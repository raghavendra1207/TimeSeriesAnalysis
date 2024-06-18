import sys

# Get the selected option from command-line arguments
if len(sys.argv) > 1:
    selected_option = sys.argv[1]
    if(selected_option=='google'):
        stock='GOOG'
    elif(selected_option=='Bitcoin'):
        stock='BTC-USD'
    elif(selected_option=='FTSE100'):
        Stock='^FTSE'
    elif(selected_option=='NIFTY 50'):
        stock='^NSEI'
    elif(selected_option=='Tesla, Inc.'):
        stock='TSLA'
else:
    print("No selected option provided.")
