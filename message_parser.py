import os
from tqdm import tqdm
from datetime import datetime

class WhatsAppParser():
    def __init__(self,
        username: str,
        group_message_allowance: int = 1200,
        answer_time_allowance: int = 10000,
        max_context_message_length: int = 200,
        debug: bool = False,
        ):
        '''
        Description:
            Parsers WhatsApp chat files and converts them into data points

        Params:
            - username (str): the username of the person to contruct the answers to the prompts around
            - group_message_allowance (int): max number of seconds allowed between messages in order for them to be treated as a single message
            - answer_time_allowance (int): max number of seconds allowed between the prevouis message and a candidate answer in order for them to be paired
            - max_context_message_length (int): the maximum number of characters allowed in a context string
            - debug (bool): if True then debugging messages (such as caught exceptions) will be printed to the terminal
        '''

        self.username = username
        self.group_message_allowance = group_message_allowance
        self.answer_time_allowance = answer_time_allowance
        self.max_context_message_length = max_context_message_length
        self.debug = debug

        # hard coded list of auto generated WhatsApp messages that should be ignored
        self.messages_to_ignore = [
            "image omitted",
            "missed voice call",
            "this message was deleted",
            "you deleted this message",
            "audio omitted",
            "gif omitted",
            "missed video call",
            "\u200e",

        ]

    def parse_folder(self, folder_path: str):
        '''
        Description:
            Parse a series of files, all located inside a folder

        Params:
            - folder_path: the global path to the folder containing the WhatsApp txt files

        Returns:
            - folder_datapoints: a list of dictionaries, each corresponding to a prompt answer datapoint
        '''

        # initialise list to store the prompt answer pairs
        folder_datapoints = []

        # only keep files that end in .txt
        valid_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        # initialise tqdm object to store valid files
        tqdm_files = tqdm(valid_files)
        tqdm_files.set_description('Parsing {0} chat files'.format(len(tqdm_files)))

        # loop over each file, attempting to parse its contents
        for file_name in tqdm_files:

            # generate global file path
            file_path = os.path.join(folder_path, file_name)

            # attempt to parse file, catch exception if parsing fails
            try:
                # parse the file
                file_datapoints = self.parse_file(file_path)

                if self.debug:
                    print(' - {0} data points were successfully generated from {1}'.format(len(file_datapoints), file_name))

                # add the files data
                folder_datapoints = folder_datapoints + file_datapoints

            except Exception as e:
                if self.debug:
                    print(' - Failed parsing {0} with exception:\n  {1}'.format(file_name, e))
            
        print('Found {0} files while parsing folder {1}, successfully generated {2} datapoint'.format(len(valid_files), folder_path, len(folder_datapoints)))

        return folder_datapoints
            
    def parse_file(self, file_path: str):
        '''
        Description:
            Parses a WhatsApp txt file

        Params:
            - file_path (str): file_path of the file to parse

        Returns:
            - datapoints (list[Dict]): a list of dictionaries, each dictionary contains a context and response data pair
        '''

        with open(file_path) as f:
            raw_lines = f.readlines()

        # process the raw chat log lines
        processed_lines = self.process_raw_lines(raw_lines)

        # combine messages sent in quick succession from the same person into one message
        grouped_messages = self.group_messages(processed_lines)

        # convert them into prompt answer pairs
        datapoints = self.generate_datapoints(grouped_messages)

        return datapoints


    def generate_datapoints(self, grouped_messages: list[dict]):
        '''
        Description:
            Convert a list of processed lines into a list of datapoints (context and response pairs)

        Params:
            - processed_lines (list[dict]): list of grouped messages

        Returns:
            - datapoints (list[dict]): a list of dictionaries, each dictionary contains information about a prompt answer pair
        '''

        # initialise datapoints list
        datapoints = []
        
        # start with the most recent message sent
        # and loop backwards to the very first
        i = len(grouped_messages) - 1
        while i > 0:
            
            # read the current message as a potential response datapoint
            response_candidate = grouped_messages[i]

            # check the author of the response message is the target username we are looking for
            if response_candidate['name'] == self.username:

                # initialise the context string
                context_string = ''

                # use this bool to flip between messages being sent by other users and the target username
                different_user = True

                # initialise j
                j = i - 1

    
                while j >= 0:

                    # read context candidate
                    context_candidate = grouped_messages[j]

                    # this if statement is to ensure that each message we add to our data point alternates between being written by the target username,
                    # and then by someone else, and then by the target username, etc...
                    # Also checks the message was sent within the time limit
                    if ((different_user and context_candidate['name'] != self.username) or (not different_user and context_candidate['name'] == (self.username)) and (context_candidate['time'] - response_candidate['time']) < self.answer_time_allowance) and len(context_string) < self.max_context_message_length:

                        # prepend the message to the front of the context string (with splitting tokens)
                        context_string = '<s> ' + context_candidate['message'] + ' </s> ' + context_string

                        # flip the different user bool
                        different_user = not different_user

                    else:

                        # if the message at the front of the context string was sent by the target user
                        # then append a dummy message to be the first message
                        if context_candidate['name'] == self.username:
                            context_string = '<s> Hello </s> ' + context_string

                        # create data point and add it to the list
                        data = {
                            'context': context_string,
                            'response': response_candidate['message']
                        }
                        datapoints.append(data)
                        
                        break

                    # if this is the last message in the whatsapp file then move to the next iteration which will add it to the datapoints list
                    # otherwise decrease j
                    if j == 0:
                        continue
                    else:
                        j -= 1

            i -= 1

        return datapoints


    def group_messages(self, processed_lines):
        '''
        Description:
            Group 'rapid' messages into a singular message, rapid messages are ones sent in quick succession by the same person (with no other messaegs in between)

        Params:
            - processed_lines (list[dict]): list of processed messages

        Returns:
            - grouped_message (list[dict]): list of grouped messages
        
        '''

        grouped_messages = []

        # start with the first message and loop to the end of the lines
        i = 0
        while i < len(processed_lines):

            # read  information about starting messages
            start_name = processed_lines[i]['name']
            start_time = processed_lines[i]['time']
            message = processed_lines[i]['message']

            # loop forward for looking for new messages to group together
            j = i + 1
            while j < len(processed_lines):

                # read a group candidate message
                candidate = processed_lines[j]

                # if group conditions are reached then group together the message strings, otherwise exit the loop
                if candidate['name'] == start_name and (candidate['time'] - start_time) < self.group_message_allowance:
                    message += '. {0}'.format(candidate['message'])
                else:
                    break

                j += 1

            # create new message dictionary and add to list
            new_message = {
                'name': start_name,
                'time': start_time,
                'message': message,
            }
            grouped_messages.append(new_message)

            # increment i to j's position, so that already grouped messages are skipped
            i = j

        return grouped_messages



    def process_raw_lines(self, raw_lines: list, print_exceptions: bool = False):
        '''
        Description:
            Takes a list of raw lines from a WhatsApp chat txt file and attempts to format each line

        Params:
            - raw_lines: list storing each line of the txt file
            - print_exceptions: if true then any time a line is unsuccessully processed the exception
                              to the terminal

        Returns:
            - processed_lines: list of dictionaries, each dictionary stores information about a single message
        '''
        
        # list to store processed lines
        processed_lines = []

        # loop over each line and attempt to extract the time, name, and message
        for line in raw_lines:

            line = line.strip()
            # attempt to read the time from the raw line
            try:
                # read time stampt, and convert it to seconds
                time_start_idx = line.find('[') + 1
                time_end_idx = line[time_start_idx:].find(']') + time_start_idx
                time = datetime.strptime(line[time_start_idx:time_end_idx], '%d/%m/%Y, %X').timestamp()
            except Exception as time_exception:
                if print_exceptions:
                    print('time reading exception: {0}'.format(time_exception))
                else:
                    continue

            # attempt to read the name from the raw line
            try:
                # read name from line
                name_start_idx = time_end_idx + 2
                name_end_idx = line[name_start_idx:].find(':') + name_start_idx
                name = line[name_start_idx:name_end_idx].lower()

                # make sure a valid name is read
                assert len(name) > 0, 'no name was found, only an empty string'

            except Exception as name_exception:
                if print_exceptions:
                    print('name reading exception: {0}'.format(name_exception))
                else:
                    continue

            # attempt to read the message from the raw line
            try:
                # read message content
                message_start_idx = name_end_idx + 2
                message = line[message_start_idx:].strip().lower()

                # make sure the message isnt empty
                assert len(message) > 0, 'no message found, only an empty string'

                # this assertion removes WhatsApp generated messages like "image omitted" or "missed call"
                # these are not real messages but notifications added to the chat by whatsapp
                # these messages are proceded by a special character, which we ignore in this assertion check by indexing with [1:]
                assert not any(substring in message.lower() for substring in self.messages_to_ignore), 'message "{0}" is in ignore message list'

            except Exception as message_exception:
                if print_exceptions:
                    print('message reading exceptions: {0}'.format(message_exception))
                else:
                    continue

            processed_lines.append({
                'time':time,
                'name':name,
                'message':message
                })

        return processed_lines


if __name__ == '__main__':

    parser = WhatsAppParser(username='joshua', debug=False)
    datapoints = parser.parse_folder('/Users/joshua/env/datasets/whatsapp_chat_logs')


    while True:

        from random import randrange
        r = randrange(len(datapoints))
        print(datapoints[r])
        import time
        time.sleep(10)
        print('\n')