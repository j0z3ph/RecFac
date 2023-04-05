from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
import requests

Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: True
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
    Button:
        id: message
        text: ''
        size_hint_y: None
        height: '48dp'
    
''')


class CameraClick(BoxLayout):
    def capture(self):

        url = 'http://localhost:3001/python'
        myobj = {}

        x = requests.post(url, json = myobj)
        data = x.json()
        
        label = self.ids['message']

        if(len(data) == 0):
            label.text = 'Usuario no identificado'
        else:
            label.text = 'Bienvenido ' + data[0]['Nombre']
            
        
        
        #camera = self.ids['camera']
        #timestr = time.strftime("%Y%m%d_%H%M%S")
        #camera.export_to_png("IMG_{}.png".format(timestr))
        #print("Captured")


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()