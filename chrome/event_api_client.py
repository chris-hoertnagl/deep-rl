import random
from paho.mqtt import client as mqtt_client

BROKER = '127.0.0.1'
PORT = 8883
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'

STATE_TOPIC = "rl/state"
ACTION_TOPIC = "rl/action"
RESET_TOPIC = "rl/reset"

# Very High level, this is going to look like the API Framework we use decides


class ApiClient():

    def __init__(self):
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
            else:
                print("Failed to connect, return code %d\n", rc)

        self.client = mqtt_client.Client(client_id, transport='websockets')
        self.client.on_connect = on_connect
        self.client.connect(BROKER, PORT)
        self.client.loop_start()

    def publish_action(self, action):
        msg = action
        result = self.client.publish(ACTION_TOPIC, msg)
        # result: [0, 1]
        status = result[0]
        return status

    def publish_reset(self):
        msg = "reset"
        result = self.client.publish(RESET_TOPIC, msg)
        # result: [0, 1]
        status = result[0]
        return status

    def subscribe_state(self, callback):
        self.client.subscribe(STATE_TOPIC)
        self.client.on_message = callback

    def close(self):
        self.client.disconnect()  # disconnect gracefully
        self.client.loop_stop()  # stops network loop
