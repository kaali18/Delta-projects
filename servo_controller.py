# servo_controller.py
import pyfirmata
import time

class ServoController:
    def __init__(self, board_port, servo_pin):
        self.board = pyfirmata.Arduino(board_port)
        self.servo_pin = servo_pin
        self.servo = self.board.get_pin('d:' + str(self.servo_pin) + ':s')

    def move_servo(self, angle):
        self.servo.write(angle)
        time.sleep(0.1)  # Add a delay to allow the servo to reach the desired position

    def cleanup(self):
        self.board.exit()

# Example usage:
# servo_controller = ServoController('COM3', 9)  # Replace 'COM3' with your Arduino port and 9 with the servo pin
# servo_controller.move_servo(90)  # Move the servo to 90 degrees
# servo_controller.cleanup()
