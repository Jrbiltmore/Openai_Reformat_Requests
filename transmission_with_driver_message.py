#jrbiltmore@icloud.com (657) 263-0133

import requests
import random
import time
import logging


class TransmissionFailedError(Exception):
    pass


class Satellite:
    def __init__(self, name, bandwidth):
        self.name = name
        self.bandwidth = bandwidth

    def transmit_data(self, data, navsat_data):
        transmission_time = len(data) / self.bandwidth
        # Simulating transmission delay
        print(f"Transmitting data through satellite {self.name}...")
        print(f"Estimated transmission time: {transmission_time} seconds")
        # Simulating transmission success or failure
        if random.random() < 0.9:  # 90% success rate
            print("Transmission successful!")
            return True
        else:
            raise TransmissionFailedError("Transmission failed!")


class CellService:
    def __init__(self, satellites):
        self.satellites = satellites

    def send_data(self, data, navsat_data, num_transmissions=1, max_retries=3, base_retry_delay=5, backoff_multiplier=2,
                  max_backoff_delay=30, timeout=30):
        successful_transmissions = 0
        failed_transmissions = 0
        total_attempts = 0
        total_retry_delay = 0

        for transmission in range(num_transmissions):
            print(f"Transmission #{transmission + 1}:")
            start_time = time.time()
            transmission_status = False
            attempts = 0

            while not transmission_status:
                attempts += 1
                total_attempts += 1

                # Selecting a random satellite for transmission
                satellite = random.choice(self.satellites)
                print(f"Transmitting data through satellite {satellite.name}...")

                try:
                    transmission_status = satellite.transmit_data(data, navsat_data)
                except TransmissionFailedError:
                    logging.exception("Transmission failed.")
                    transmission_status = False

                if transmission_status:
                    successful_transmissions += 1
                else:
                    failed_transmissions += 1
                    if time.time() - start_time >= timeout:
                        print("Transmission timeout!")
                        break

                    if attempts <= max_retries:
                        retry_delay = self.calculate_retry_delay(base_retry_delay, backoff_multiplier, attempts,
                                                                 max_backoff_delay)
                        total_retry_delay += retry_delay
                        print(f"Retrying transmission in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print("Data transmission failed after maximum retries.")
                        break

        print(f"\nTransmission Statistics:")
        print(f"Successful Transmissions: {successful_transmissions}")
        print(f"Failed Transmissions: {failed_transmissions}")
        print(f"Total Attempts: {total_attempts}")
        print(f"Total Retry Delay: {total_retry_delay} seconds")

    @staticmethod
    def calculate_retry_delay(base_retry_delay, backoff_multiplier, attempts, max_backoff_delay):
        # Calculate exponential backoff with capped delay
        retry_delay = min(base_retry_delay * (backoff_multiplier ** (attempts - 1)), max_backoff_delay)
        return retry_delay


class DriverMessage:
    def __init__(self, message):
        self.message = message

    def send(self):
        # Implement the logic to send the message via SMS
        # ...
        try:
            response = requests.post(SMS_API_URL, json={"message": self.message})
            response.raise_for_status()  # Raise an exception for non-2xx response codes
            return True
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send driver message: {str(e)}")
            return False


def get_top_satellite_downlink():
    """Gets the top satellite downlink.

    Returns:
        The top satellite downlink.
    """
    url = "https://api.spacexdata.com/v4/telemetry/downlinks/top"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def main():
    """Performs data transmission and allows the driver to send a message."""

    # Get top satellite downlink
    downlink = get_top_satellite_downlink()
    if downlink:
        print(f"The top satellite downlink is {downlink['name']}.")
    else:
        print("No top satellite downlink found.")

    # Create satellite instances
    satellite1 = Satellite("Satellite1", 100)  # Name and bandwidth in Mbps
    satellite2 = Satellite("Satellite2", 200)
    satellite3 = Satellite("Satellite3", 150)

    # Create cell service instance with available satellites
    cell_service = CellService([satellite1, satellite2, satellite3])

    # Send data through the cell service
    data = "Hello, world!"
    navsat_data = "NavSat data"
    cell_service.send_data(data, navsat_data, num_transmissions=5, max_retries=3, base_retry_delay=5,
                           backoff_multiplier=2, max_backoff_delay=30, timeout=30)

    # Prompt the driver to send a message
    driver_message = input("Enter a message to send: ")
    driver_message_obj = DriverMessage(driver_message)
    if driver_message_obj.send():
        print("Driver message sent successfully!")
    else:
        print("Failed to send driver message.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(filename='transmission.log', level=logging.ERROR)

    # Execute the main function
    main()
