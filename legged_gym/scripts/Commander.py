class GamepadCommander(object):
    def __init__(self, gamepad_type='Xbox'):
        from legged_gym.gamepad import gamepad, controllers
        if not gamepad.available():
            raise EOFError('Gamepad not found')
        try:
            self.gamepad: gamepad.Gamepad = getattr(controllers, gamepad_type)()
            self.gamepad_type = gamepad_type
        except AttributeError:
            raise RuntimeError(f'`{gamepad_type}` is not supported, '
                               f'all {controllers.all_controllers}')
        self.gamepad.startBackgroundUpdates()
        print('Gamepad connected')

    @classmethod
    def is_available(cls):
        from legged_gym.gamepad import gamepad
        return gamepad.available()

    def read_command(self):
        if self.gamepad.isConnected():
            x_speed = -self.gamepad.axis('LAS -Y')
            y_speed = -self.gamepad.axis('LAS -X')
            steering = -self.gamepad.axis('RAS -X')
            return x_speed, y_speed, steering
        else:
            raise EOFError('Gamepad disconnected')

    def __del__(self):
        self.gamepad.disconnect()

    @property
    def normalized(self):
        return True
