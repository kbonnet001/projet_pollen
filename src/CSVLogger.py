import csv
import os

class CSVLogger:
    def __init__(self, filename, headers):
        self.filename = filename
        self.headers = ["Iteration"] + headers  # Ajoute "Iteration" automatiquement
        self.file_exists = os.path.exists(filename)

        if not self.file_exists:
            with open(self.filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def get_next_iteration(self):
        """Read last iteration and add name"""
        if not self.file_exists:
            return 0
        with open(self.filename, mode="r") as file:
            rows = list(csv.reader(file))
            if len(rows) > 1:  
                return int(rows[-1][0]) + 1
        return 0  

    def log(self, data):
        """ add new line."""
        i = self.get_next_iteration()
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i] + data)

