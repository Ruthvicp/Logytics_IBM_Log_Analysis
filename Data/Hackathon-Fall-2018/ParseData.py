import collections
import re

FILE_OUT = 'window_data_no_slide.txt'
FILE_IN = 'worker.data'
WINDOW_SIZE = 25

# Fixed size queue and a list for holding data for the sliding window code
Window_Data = collections.deque(WINDOW_SIZE * [""], WINDOW_SIZE)
Extra_Data = []
FLAG = True

file_out = open(FILE_OUT, 'w')

# file_out.write("log,stream,time\n") # Writes headers for CSV


# splits the log file and filters out special characters
# returns a comma separated string
def csvFull(line):
    dataList = line.split("\",\"")
    log = dataList[0].split(":", 1)

    # filters out dates, special characters, and time from log message
    # also converts any number after filtering into a capital A
    message = re.sub(r'[\[\]{}"]', '', str(log[1]))
    message = message.replace(r"\n", "").replace(r"\u003c", "").replace(r"\u003e", "").replace(r"\u0026", " ") \
        .replace("\\", "").replace("-", " ").replace("$", "")
    message = re.sub(r'[:/_*=.#%|+]', ' ', message)
    message = re.sub(r'["?()@]', '', message)
    message = re.sub('([1-9][0-9][0-9][0-9])(-)([1-9][0-9])(-)([0-9][0-9])', '', message)
    message = re.sub('([0-2][0-9]):([0-5][0-9]):([0-5][0-9])', '', message)
    message = re.sub('[.][0-9][0-9][0-9]', '', message)
    message = re.sub('[+][0-9][0-9][0-9][0-9]', '', message).strip()
    message = message.replace(" ", "_")
    message = re.sub('[0-9]', 'A', message)

    streamData = dataList[1].split(":")
    stream = re.sub('[{}"]', '', str(streamData[1])).strip()

    timeData = dataList[2].split(":", 1)
    time = re.sub('[{}"]', '', str(timeData[1]))
    return str(message) + "," + stream + "," + time.strip() + "\n"


# performs a sliding window over the data and writes the data to a file
def sliding():
    global FLAG
    window = []
    Extra_Data.reverse()
    for x in range(0, len(Window_Data)):
        window.append(Window_Data.pop())
    for x in range(0, len(Extra_Data)):
        window.append(Extra_Data.pop())
    '''
    for i in range(len(window)):
        res = window[i : i + WINDOW_SIZE]
        if len(res) == WINDOW_SIZE:
            for x in res:
                if x != "":
                    file_out.write(x + ". ")'''
    for x in window:
        file_out.write(x)

    file_out.write("\n")
    FLAG = True


# adds a log message to a queue or list based on FLAG after removing special characters and date/time
def addQueue(line):
    dataList = line.split("\",\"")
    log = dataList[0].split(":", 1)

    message = re.sub(r'[\[\]{}"]', '', str(log[1]))
    message = message.replace(r"\n", "").replace(r"\u003c", "<").replace(r"\u003e", ">").replace(r"\u0026", "&") \
        .replace("\\", "")
    message = re.sub('([1-9][0-9][0-9][0-9])(-)([1-9][0-9])(-)([0-9][0-9])', '', message)
    message = re.sub('([0-2][0-9]):([0-5][0-9]):([0-5][0-9])', '', message)
    message = re.sub('[.][0-9][0-9][0-9]', '', message)
    message = re.sub('[+][0-9][0-9][0-9][0-9]', '', message).strip()
    message = re.sub('[0-9]', 'A', message)

    if (FLAG):
        Window_Data.appendleft(message)
    else:
        Extra_Data.append(message)


# Finds if a log message has an error message in it and changes the FLAG value
def checkInput(line):
    global FLAG
    dataList = line.split("\",\"")
    log = dataList[0].split(":", 1)
    message = str(log[1])

    addQueue(line)

    if (message.find("ERR") != -1):
        FLAG = False
    elif (message.find("Error") != -1):
        FLAG = False
    elif (message.find("ERROR") != -1):
        FLAG = False


# Reads the file
with open(FILE_IN, 'r') as file_in:
    for line in file_in:

        if not FLAG:
            if len(Extra_Data) == WINDOW_SIZE:
                sliding()

        checkInput(line)

        ##file_out.write(csvFull(line))

file_out.close()
