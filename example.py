import multiprocessing
import os
import requests
import time

if __name__ == '__main__':
    # Avvia il server Flask in un processo separato
    server_process = multiprocessing.Process(target=os.system, args=('python3 textpreprocessed2moral.py',))
    server_process.start()

    # Aspetta che il server Flask sia avviato
    time.sleep(5)  # Attendi alcuni secondi per avviare il server

    url = "http://127.0.0.1:5000"

    data = {
        'text': {'moral': ': noi, che accogliamo i profughi in famiglia in casa nostra', 'topic': 'accogliere profugo famiglia'}
        }

    def request_and_print():
        response = requests.get(f"{url}/moral_prediction", json=data)
        print(response.status_code)
        print(response.json())

    request_and_print()


