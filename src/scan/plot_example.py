#gru command for exp2
gru_acc = [18.16,8.98,16.07,9.26,20.01]
gru_oracle = [58.65,40.05,59.54,54.72,60.02]
gc_1={"command_length": { "4": 0.0, "7": 0.0, "6": 0.0,"8": 0.1640625, "9": 0.2751196172248804},"action_length": {"24": 0.9613095238095238, "25": 0.5558035714285714, "26": 0.0, "27": 0.0, "28": 0.11383928571428571, "30": 0.0, "32": 0.19866071428571427, "33": 0.0, "36": 0.0, "40": 0.0, "48": 0.0}}
gc_2={"command_length": {"4": 0.0, "7": 0.0, "6": 0.0,"8": 0.1171875, "9": 0.10287081339712918},"action_length": {"24": 0.5952380952380952, "25": 0.3392857142857143, "26": 0.0, "27": 0.0, "28": 0.0, "30": 0.0, "32": 0.0, "33": 0.0, "36": 0.0, "40": 0.0, "48": 0.0}}
gc_3 = {"command_length": {"4": 0.0, "7": 0.0, "6": 0.0,"8": 0.154296875, "9": 0.23504784688995214,},"action_length": {"24": 0.9017857142857143, "25": 0.5580357142857143, "26": 0.0, "27": 0.0, "28": 0.08482142857142858, "30": 0.0, "32": 0.08705357142857142, "33": 0.0, "36": 0.0, "40": 0.0, "48": 0.0}}
gc_4 = {"command_length": { "4": 0.0, "7": 0.0, "6": 0.0,"8": 0.119140625, "9": 0.1076555023923445},"action_length": {"24": 0.6011904761904762, "25": 0.33035714285714285, "26": 0.0, "27": 0.0, "28": 0.017857142857142856, "30": 0.001736111111111111, "32": 0.008928571428571428, "33": 0.0, "36": 0.0, "40": 0.0, "48": 0.0}}
gc_5 = {"command_length": {"4": 0.0, "7": 0.0, "6": 0.0,"8": 0.179140625, "9": 0.2876555023923445},"action_length": {"24": 0.7011904761904762, "25": 0.63035714285714285, "26": 0.0, "27": 0.0, "28": 0.017857142857142856, "30": 0.001736111111111111, "32": 0.008928571428571428, "33": 0.0, "36": 0.0, "40": 0.0, "48": 0.0}}

gc_command = np.array([np.array(list(gc_1["command_length"].values())),np.array(list(gc_2["command_length"].values())),np.array(list(gc_3["command_length"].values())),np.array(list(gc_4["command_length"].values())),np.array(list(gc_5["command_length"].values()))])
gc_action = np.array([np.array(list(gc_1["action_length"].values())),np.array(list(gc_2["action_length"].values())),np.array(list(gc_3["action_length"].values())),np.array(list(gc_4["action_length"].values())),np.array(list(gc_5["action_length"].values()))])

gc_command = np.array([np.array(list(gc_1["command_length"].values())),np.array(list(gc_3["command_length"].values())),np.array(list(gc_5["command_length"].values()))])
gc_action = np.array([np.array(list(gc_1["action_length"].values())),np.array(list(gc_3["action_length"].values())),np.array(list(gc_5["action_length"].values()))])



width = total_width, n = 0.8, 2
width = total_width/n

plt.bar(np.array([4,6,7,8,9])- width/2,np.mean(gc_command,axis=0)*100,width,yerr = np.std(gc_command,axis=0)/np.sqrt(5)*100,label='GRU')

#add more bars
# plt.bar(np.array([4,6,7,8,9])+ width/2,np.mean(lstm_command,axis=0)*100,width,yerr = np.std(lstm_command,axis=0)/np.sqrt(5)*100,label='LSTM')

plt.ylabel('Accuracy on New Commands%')
plt.xlabel('Command Length')
plt.legend()
plt.savefig('command.png',dpi=600)
