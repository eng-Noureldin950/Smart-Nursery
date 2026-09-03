import tkinter as tk

class SmartNurseryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Nursery Guardian")
        self.root.geometry("650x350")
        
        # here I Setup main UI components
        self.status_label = tk.Label(root, text="System Calm", font=("Arial", 20))
        self.status_label.pack(pady=40)
        
        self.temp_label = tk.Label(root, text="Temp: ", font=("Arial", 16))
        self.temp_label.pack(pady=10)

        # Testing control panel (we have to  remove it later)
        self.test_frame = tk.Frame(self.root)
        self.test_frame.pack(side=tk.BOTTOM, pady=30)
        # Just buttons I have put until complteting our model and telegram bot 
        tk.Button(self.test_frame, text="Test Gas", command=self.trigger_gas_alert, bg="red", fg="white", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.test_frame, text="Test Hungry", command=lambda: self.update_gui("Hungry Cry: Playing Video", "orange"), font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.test_frame, text="Test Tired", command=lambda: self.update_gui("Tired Cry: URGENT!", "red"), font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.test_frame, text="Test High Temp (32°C)", command=lambda: self.update_temp_display(32), font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(self.test_frame, text="Reset", command=self.reset_gui, font=("Arial", 12), bg="lightgray").pack(side=tk.LEFT, padx=5)

    def trigger_gas_alert(self):
        # Full screen red alert for safety
        self.root.config(bg="red")
        self.status_label.config(bg="red", fg="white", text="SAFETY ALERT: GAS DETECTED!")
        self.temp_label.config(bg="red", fg="white")
        self.test_frame.config(bg="red")

    def update_gui(self, text, color):
        self.status_label.config(text=text, fg=color)

    def update_temp_display(self, temp):
        # Temperature color coding logic depending on T('c)
        if temp < 25:
            color = "blue"
        elif 25 <= temp <= 30:
            color = "green"
        else:
            color = "red"
        self.temp_label.config(text=f"Temp: {temp}°C", fg=color)

    def reset_gui(self):
        # Resets to default state
        default_bg = "SystemButtonFace"
        self.root.config(bg=default_bg)
        self.test_frame.config(bg=default_bg)
        self.status_label.config(text="System Calm...", fg="black", bg=default_bg)
        self.temp_label.config(text="Temp: --", fg="black", bg=default_bg)

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartNurseryApp(root)
    root.mainloop()