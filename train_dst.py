def train(epochs=30, lr = 0.0003): 
    for epoch in range(0,100):
        for i, (images, img_names, idx) in enumerate(tqdm(train_loader)):
            loss.backward()
            optimizer.step()

            
                
        scheduler.step()
        
        train_loss = batch_loss/train_batch_s
        print("Epoch Training loss: ", train_loss) #batch_size denominator 32

        print("\n")
        rec_results = recognition_metrics(decoded_preds, decoded_labels, file_name="results.csv")
        print('\n')
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
        
        try:    
            report = classification_report(y_true, y_pred, labels=np.arange(1, len(inv_grapheme_dict)+1), 
                                           zero_division=0, output_dict=True, target_names=[v for k, v in inv_grapheme_dict.items()])
            f1_micro = f1_score(y_true, y_pred, average = 'micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average = 'macro', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)

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

        scheduler.step()
        torch.save(student_model.state_dict(), './'+path+'/epoch{}.pth'.format(epoch))
        