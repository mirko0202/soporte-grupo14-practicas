from kivy.app import App
from kivy.lang import Builder

class CalculadoraApp(App):
    def build(self):
        return Builder.load_file('calculadora.kv')
        
if __name__ == '__main__':
    CalculadoraApp().run()