import os
from tqdm import tqdm
from datetime import datetime

class WhatsAppParser():
    def __init__(self,
        username: str,
        prompt_time_allowance: int = 300,
        answer_time_allowance: int = 10000,
        debug: bool = False,
        ):
        '''
        Description:
            Class for parsing WhatsApps chat files

        Params:
            - username: the username of the person to contruct the answers to the prompts around
            - prompt_time_allowance: max number of seconds allowed between messages in order for them to be chained together in a prompt
            - answer_time_allowance: max number of seconds allowed between the final message of a prompt and a candidate answer in order for them to be paired
            - debug: if True then debugging messages (such as caught exceptions) will be printed to the terminal
        '''

        self.username = username
        self.prompt_time_allowance = prompt_time_allowance
        self.answer_time_allowance = answer_time_allowance
        self.debug = debug

        # hard coded list of auto generated WhatsApp messages that should be ignored
        self.messages_to_ignore = [
            "image omitted",
            "Missed voice call",
            "This message was deleted.",
            "You deleted this message",
            "audio omitted",
            "GIF omitted",
            "Missed video call",
            "Messages and calls are end-to-end encrypted. No one outside of this chat, not even WhatsApp, can read or listen to them.",
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
            - file_path: file_path of the file to parse

        Returns:
            - datapoints: a list of dictionaries, each dictionary contains information about a prompt answer pair
        '''

        with open(file_path) as f:
            raw_lines = f.readlines()

        # process the raw chat log lines
        processed_lines = self.process_raw_lines(raw_lines)

        # convert them into prompt answer pairs
        datapoints = self.generate_datapoints(processed_lines)

        return datapoints


    def generate_datapoints(self, processed_lines: list):
        '''
        Description:
            Convert a list of processed lines into a list of datapoints (prompt answer pairs)

        Params:
            - processed_lines: list of processed text lines

        Returns:
            - datapoints: a list of dictionaries, each dictionary contains information about a prompt answer pair
        '''

        # start at line 0 of the text file
        index = 0

        # initialise datapoints list
        datapoints = []
        
        # loop through the series of lines, we exit the while loop when index <= num lines - 2 because we require
        # at least two messages to generate a text answer pair, with this exit clause we attempt to make a prompt
        # out of all but the last message sent in the file
        while index <= len(processed_lines) - 2:
            
            # attempt to chain together a series of messages into a prompt
            prompt, prompt_end_index = self.chain_message(index, processed_lines)

            # attempt to find an answer to the prompt
            answer = self.generate_answer(prompt, processed_lines)

            # if answer is found and the name matches the specified username then add it to datapoints
            if answer and answer['name'] == self.username:
                datapoints.append({
                    'prompt':prompt,
                    'answer':answer
                })

            # set the next prompt index to attmpt to start just after the current processed prompt
            index = prompt_end_index

        return datapoints

    def generate_answer(self, prompt, processed_lines):
        '''
        Description:
            Given a prompt, attempts to generate a matching answer

        Params:
            - prompt: dictionary object containing prompt information
            - processed_lines: list of processed lines

        Returns:
            - answer: dictionary containing information from the generated answer
            OR
            - None: if no answer could be found then return None
        '''

        # start searching for an answer from the end of the prompt index + 1
        answer_start_index = prompt['end_index'] + 1
        candidate_answer = processed_lines[answer_start_index]

        # if the message was written by someone else and the message time was within the allowed limit 
        if (candidate_answer['name'] != prompt['name']) and (prompt['end_time'] - candidate_answer['time'] < self.answer_time_allowance):
            answer, _ = self.chain_message(answer_start_index, processed_lines)
            return answer
        else:
            return None

    def chain_message(self, prompt_index, processed_lines):

        # read the start of the prompt
        prompt_start = processed_lines[prompt_index]

        prompt = {
            'name': prompt_start['name'],
            'start_index': prompt_index,
            'start_time': prompt_start['time']
        }

        # initialise the prompt message
        prompt_message = '{0}.'.format(prompt_start['message'])

        if not prompt_message.endswith('.'):
            prompt_message = prompt_message + '.'

        # initialise prompt current
        prompt_current = prompt_start

        # keep looping until the end of the processed lines if nessassary
        while prompt_index < len(processed_lines) - 1:
            # increment index
            prompt_index += 1

            # read the next message
            candidate_prompt_addition = processed_lines[prompt_index]

            # test if the new message fits the requirements for being added to the current prompt
            if (candidate_prompt_addition['name'] == prompt_current['name']) and ((candidate_prompt_addition['time'] - prompt_current['time']) < self.prompt_time_allowance):

                # if requirements are met then append the candidates message on to the prompts message
                prompt_message = prompt_message + ' ' + candidate_prompt_addition['message']

                # add a full stop if nessassary
                if not prompt_message.endswith('.'):
                    prompt_message = prompt_message + '.'

                # update prompt current, used to maintain time checks in the next cycle of the while loop
                prompt_current = processed_lines[prompt_index]

            else:
                break

        # set the end index to be the prompt index - 1 becuase the final index from the while loop would have failed to chain
        prompt['end_index'] =  prompt_index - 1
        prompt['end_time'] = prompt_current['time']
        prompt['message'] = prompt_message

        # if conditions are not met then stop the prompt 'chain' and return the prompt message and the end index of the prompt
        return prompt, prompt_index

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
                assert not any(substring in message for substring in self.messages_to_ignore), 'message "{0}" is in ignore message list'

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
