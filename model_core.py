class RobertaForPromptFinetuning(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
        dep_input_ids=None,
        dep_attention_mask=None,
        dep_token_type_ids=None,
        dep_mask_pos=None,
        pos_input_ids=None,
        pos_attention_mask=None,
        pos_token_type_ids=None,
        pos_mask_pos=None,
        neg_input_ids=None,
        neg_attention_mask=None,
        neg_token_type_ids=None,
        neg_mask_pos=None,
            # 如果多个负样例 构造一个负样例数组的 input_ids
    ):
        batch_size = input_ids.size(0)
        if self.data_args.use_dependency_template & (self.data_args.use_compare_lm is None):
            input_ids = dep_input_ids
            mask_pos = dep_mask_pos
            attention_mask = dep_attention_mask

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            if self.data_args.use_dependency_template & (self.data_args.use_compare_lm is not None):
                dep_mask_pos = dep_mask_pos.squeeze()
                pos_mask_pos = pos_mask_pos.squeeze()
                neg_mask_pos = neg_mask_pos.squeeze()
        # Encode everything
        outputs = self.roberta(
            input_ids,  # [2,128]
            attention_mask=attention_mask
        )

        # Get <mask> token representation  [2,128,1024]  [2,1024]
        sequence_output, pooled_output = outputs[:2]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos]


        if self.data_args.use_compare_lm == "dep_positive":
            dep_outputs = self.roberta(
                dep_input_ids, # [2,128]
                attention_mask=dep_attention_mask
            )
            # Get <mask> token representation of positive
            dep_sequence_output, dep_pooled_output = dep_outputs[:2]
            dep_sequence_mask_output = dep_sequence_output[torch.arange(dep_sequence_output.size(0)), dep_mask_pos]

            pos_outputs = self.roberta(
                pos_input_ids,
                attention_mask=pos_attention_mask
            )
            pos_sequence_output, pos_pooled_output = pos_outputs[:2]
            #【2，1024】
            pos_sequence_mask_output = pos_sequence_output[torch.arange(pos_sequence_output.size(0)), pos_mask_pos]
            t = 0.07
            # Get <mask> token representation  [2,4,128,1024]  [2,4,1024]
            #neg_input_ids : [2,4,128]
            neg_sequence_output = [list() for i in range(neg_input_ids.size(1))]
            neg_pooled_output = [list() for i in range(neg_input_ids.size(1))]
            neg_sequence_mask_output = [list() for i in range(neg_input_ids.size(1))]
            neg_sequence_mask_output_reverse = [list() for i in range(neg_input_ids.size(1))]
            all_negativeLogit = []
            for idx in range(len(neg_input_ids[0])):
                sample_neg_outputs = self.roberta(
                    neg_input_ids[:, idx, :], # [2,128]
                    attention_mask = neg_attention_mask[:, idx, :]) # [2,128，1024]
                neg_sequence_output[idx], neg_pooled_output[idx] = sample_neg_outputs[:2]
                neg_sequence_mask_output[idx] = neg_sequence_output[idx][torch.arange(neg_sequence_output[idx].size(0)), neg_mask_pos.T[idx]]
                neg_sequence_mask_output_reverse[idx] = neg_sequence_output[idx][torch.arange(neg_sequence_output[idx].size(0)), neg_mask_pos.T[idx]]
                #all_negativeLogit.append(torch.div(torch.matmul(sequence_mask_output,neg_sequence_mask_output[idx].T),t))
                # if sample_idx == len(neg_input_ids):
                #     all_neg_sequence_mask_output.sppend(neg_sequence_mask_output)
            neg_sequence_mask_output = torch.stack(neg_sequence_mask_output).permute([1,0,-1])  # [2,4,1024]
            neg_sequence_mask_output_reverse = torch.stack(neg_sequence_mask_output_reverse).permute([1,0,-1])
            #neg_sequence_mask_output_reverse = neg_sequence_mask_output  # copy to reverse, or it will chnage in next step
            #neg_sequence_mask_output = torch.div(torch.matmul(neg_sequence_mask_output,sequence_mask_output.unsqueeze(-1)), t)  # [2,4,1024][2,1024,1]
            neg_sequence_mask_output = torch.div(nn.CosineSimilarity(dim=-1)(sequence_mask_output.unsqueeze(1).expand(-1, len(neg_input_ids[1]), -1),
                                                                  neg_sequence_mask_output), t)

            neg_sequence_mask_output_reverse = torch.div(
                nn.CosineSimilarity(dim=-1)(pos_sequence_mask_output.unsqueeze(1).expand(-1, len(neg_input_ids[1]), -1),
                                            neg_sequence_mask_output_reverse), t)
            # neg_sequence_mask_output_reverse = torch.div(
            #     nn.CosineSimilarity(dim=-1)(pos_sequence_mask_output.unsqueeze(1).expand(-1, 4, -1),
            #                                 neg_sequence_mask_output), t)

        #positiveLogit = torch.div(torch.matmul(dep_sequence_mask_output, pos_sequence_mask_output.T),t)
        positiveLogit = torch.div(nn.CosineSimilarity(dim=1)(sequence_mask_output,pos_sequence_mask_output),t)
        #positiveLogit = torch.sum(positiveLogit * torch.eye(batch_size).cuda(), dim=1)
        
        logits_max = torch.max(torch.max(torch.max(neg_sequence_mask_output,neg_sequence_mask_output_reverse)),torch.max(positiveLogit))
       
        neg_sequence_mask_output = neg_sequence_mask_output - logits_max.detach()
        neg_sequence_mask_output_reverse = neg_sequence_mask_output_reverse - logits_max.detach()
        positiveLogit = positiveLogit - logits_max.detach()

        finalExplogit = torch.div(torch.exp(positiveLogit),torch.exp(positiveLogit)+torch.exp(torch.sum(neg_sequence_mask_output,dim=1).squeeze(-1))) #这里按照不调换i和j位置的输出 pos只有1条 所以都一样
        finalExplogit_reverse = torch.div(torch.exp(positiveLogit),torch.exp(positiveLogit)+torch.exp(torch.sum(neg_sequence_mask_output_reverse,dim=1).squeeze(-1))) #这里调换i和j位置的输出
        t = 1
        compareLogit = -torch.log(finalExplogit) - t * torch.log(finalExplogit_reverse) # 两部分合起来 如要多个损失 则重写这一部分
        compareLoss = torch.sum(compareLogit)
        # a = torch.eye(2)
        # b = torch.div(torch.matmul(pos_sequence_mask_output, dep_sequence_mask_output.T), t)
        # torch.mm(a, b)
        # maskpositiveLogit = positiveLogit * mask
        #negativeLogit = torch.div(torch.matmul(dep_sequence_mask_output,pos_sequence_mask_output.T),T)
        # positveExpLogit = torch.exp(positiveLogit)
        # negativeExpLogit = torch.exp(negativeLogit)
        # for each_negative_output in neg_sequence_mask_output:
        #     negativeLogit = torch.exp((torch.div(torch.matmul([dep_sequence_mask_output * len(neg_sequence_mask_output)], each_negative_output.T),T)))

        #positive_sim = dep_sequence_mask_output * sequence_mask_output
        
        #cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        #positive_result = cos(dep_sequence_mask_output,pos_sequence_mask_output)/T

        #negative_result = cos(sequence_mask_output,pos_sequence_mask_output)/T+cos(sequence_output,neg_sequence_mask_output)/T

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output) #[2,50264]

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            #print(prediction_mask_scores[:, self.label_word_list[label_id]])
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1)) # ==> [2,1]或[1,-1]  logits : 5{[2,1]}
        #logit append多了几个之后 就算cat一个或者多个都可以
        logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            logsoftmax = nn.LogSoftmax(-1)
            logits = logsoftmax(logits) # Log prob of right polarity

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        tep = self.data_args.temperature
        loss = loss + tep * compareLoss
        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output