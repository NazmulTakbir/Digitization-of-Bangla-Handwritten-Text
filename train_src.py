def train(epochs=30, lr = 0.0003): #lr=0.001
    
    """
    During Quantization Aware Training, the LR should be 1-10% of the LR used for training without quantization
    """
    
    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr, momentum=0.9, dampening=0, w

    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    # new_lr = lr * factor
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2, threshold=0.01, min_lr=0.0001, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 25], gamma=0.5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1, eta_min=0.0001)

    #early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(0,100):
        
        student_model.train()

        #necessary variables
        y_true = []
        y_pred = []
        pred_ = []
        label_ = []
        decoded_preds = []
        decoded_labels = []

        total_wer = 0

        print("***Epoch: {}***".format(epoch))
        batch_loss = 0
        #lr = 0.002
        # t = 2
        alpha = 0.5
        for i, (inp, img_names, idx) in enumerate(tqdm(train_loader)):
            
          
            inp = inp.to(device)
            inp = inp.float()/255.
            batch_size = inp.size(0)
            idxs = idx.detach().numpy()
            img_names = list(img_names)
            words, labels, labels_size = get_padded_labels(idxs, words_tr, labels_tr, lengths_tr)
            
            #print(inp.shape)
            
            teacher_preds = pred_teacher(teacher_model,labels)
            teacher_preds = teacher_preds.cuda()
        
            #teacher_preds = teacher_model(inp)
            #teacher_preds.to(device) 
            #print(teacher_preds.shape)
            #print(teacher_preds[3][0][0])
            # soft_targets = F.log_softmax(teacher_preds /3)
            #print(soft_targets.shape)
            #print(teacher_preds)
         
            z_score = student_model(inp)
         
            preds = torch.nn.functional.log_softmax(z_score , dim=2)
            pr = torch.nn.functional.log_softmax(z_score/t , dim=2)
            #preds = torch.nn.functional.log_softmax(model(inp), dim=2)
            labels = torch.tensor(labels, dtype=torch.long)
            labels.to(device)
            labels_size = torch.tensor(labels_size, dtype=torch.long)
            labels_size.to(device)
            preds_size = torch.tensor([preds.size(0)] * batch_size, dtype=torch.long)
            preds_size.to(device)
            #print(preds[3][0])
            #np_arr = preds.cpu().detach().numpy()
            #preds = torch.from_numpy(np_arr)
            
            #print(preds.shape)
            #print(preds[3][0][0])
            # soft_preds = F.log_softmax(preds / t)
            #print(soft_preds.shape)
            
            #kl_div_loss = F.kl_div(soft_preds, soft_targets)
            #kl_div_loss = 
            
            #print(kl_div_loss)
   
            
            #loss = criterion(preds, labels, preds_size, labels_size)
            
            ctc_loss = criterion(preds, labels, preds_size, labels_size)
    
            #loss =  nn.KLDivLoss()(pr , teacher_preds) * ( 1- alpha)  + ctc_loss * alpha
            ty = nn.KLDivLoss()(pr , teacher_preds)
            #ty = ty.cuda()
            loss = ty* (t*t * 2.0 + alpha) + ctc_loss *(1.-alpha)
            #print(loss.item())
            batch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().detach().cpu().numpy()
            labels = labels.detach().numpy()
        
            
            
            for pred, label in zip(preds, labels):
                decoded, _ = decode_prediction(pred, inv_grapheme_dict)
                #_, decoded_label_ = decode_prediction(labels[i], inv_grapheme_dict, gt=True)
                for x, y in zip(decoded, label):
                    
                    y_pred.append(x)
                    
                
                    y_true.append(y)
                    
                _, decoded_pred = decode_prediction(pred, inv_grapheme_dict)
                _, decoded_label = decode_prediction(label, inv_grapheme_dict, gt=True)
                # print("Pred: " + decoded_pred)
                # print("Truth: " + decoded_label)
                
                decoded_preds.append(decoded_pred)
                decoded_labels.append(decoded_label)
            #print(pred_)
                
        scheduler.step()
        
        train_loss = batch_loss/train_batch_s
        print("Epoch Training loss: ", train_loss) #batch_size denominator 32

        print("\n")
        rec_results = recognition_metrics(decoded_preds, decoded_labels, file_name="results.csv")
        print('\n')
        #print(pred_)
        #print(label_)
        print("Absolute Word Match Count: %d" % rec_results['abs_match'])
        print("Word Recognition Rate (WRR): %.4f" % rec_results['wrr'])
        print("Normal Edit Distance (NED): %.4f" % rec_results['total_ned'])
        print("Character Recognition Rate (CRR): %.4f" % rec_results['crr'])
        print("\n")
        print("End of Epoch ",epoch)
        print("\n\n")
        
        with open("./"+path+"/results_training.txt", 'w') as fout:
            for x, y in zip(pred_, label_):
                fout.write("True: {}".format(y))
                fout.write("\n")
                fout.write("Pred: {}".format(x))
                fout.write("\n\n")
    
        metrics = pd.DataFrame([{'epoch': epoch,
                                 'crr': rec_results['crr'],
                                 'wrr': rec_results['wrr'],
                                 'ned': rec_results['total_ned'],
                                 'abs_match': rec_results['abs_match'],
                                 'train_loss': train_loss
                                 }])
        
        metrics.to_csv("./"+path+"/metrics_training.csv", 
                       mode=('w' if epoch==0 else 'a'), index=False, header=(True if epoch==0 else False))
        
        
        #total_wer, _ = compute_wer(pred_, label_)
        #print("Total AED Word Error Rate (Training): %.4f" % total_wer)

        #change in number of labels
        try:    
            report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
                                           zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
            f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
        #change in number of labels


            #Absolute word matching
            #abs_correct = absolute_word_match(pred_, label_)

            #print("Absolute Word Match Count: {}".format(abs_correct))
            #print("Absolute Word Match Percentage: %.4f" % (abs_correct / num_train_samples))
            print("Training Accuracy: %.4f" % accuracy)
            print("Training F1 Micro Score: %.4f" % f1_micro)
            print("Training F1 Macro Score: %.4f" % f1_macro)
            print("\n")
            print("End of Training FOR Epoch {}".format(epoch))
            print("\n\n")

            ##################### Generate Training Report ##############################

            with pd.ExcelWriter("./"+path+"/classification_report_training.xlsx", engine='openpyxl', mode=('w' if epoch==0 else 'a')) as writer:  
                pd.DataFrame(report).T.sort_values(by='support', ascending=False).to_excel(writer, sheet_name='epoch{}'.format(epoch))

            with open("./"+path+"/results_training.txt", 'w') as fout:
                for x, y in zip(pred_, label_):
                    fout.write("True: {}".format(y))
                    fout.write("\n")
                    fout.write("Pred: {}".format(x))
                    fout.write("\n\n")

            metrics = pd.DataFrame([{'epoch': epoch,
                                    'accuracy': accuracy,
                                    'train_loss': train_loss, 

                                    'f1_micro': f1_micro, 
                                    'f1_macro': f1_macro

                                    }])

        except:
            print('error')
        #############################################################################'''
        
        valid_loss = validate(epoch, valid_batch_s, train_loss)
        #early_stopping(valid_loss, student_model)
        
        #if early_stopping.early_stop:
            #print("Early stopping")
            #break
        scheduler.step()
        torch.save(student_model.state_dict(), './'+path+'/epoch{}.pth'.format(epoch))
        