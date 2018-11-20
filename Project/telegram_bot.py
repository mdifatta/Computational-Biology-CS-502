import requests
#https://api.telegram.org/bot706008982:AAEiouZ1AoLJdtL_YIlQNo-vnd5DuCZRMkM/getUpdates
token = '706008982:AAEiouZ1AoLJdtL_YIlQNo-vnd5DuCZRMkM'

class TelegramBot:
    def __init__(self,chat_id='-266321188'):
        '''
        :param chat_id: is the chat id for the people you are interested in, you can find it at the reference link.
        '''
        self.reference_link = 'https://api.telegram.org/bot'+token+'/getUpdates'
        self.chat_id = chat_id

    def send_message(self,text):
        print(text)
        message=requests.utils.quote(text)
        requests.get('https://api.telegram.org/bot'+token+'/sendMessage?chat_id='+self.chat_id+'&text='+message)

