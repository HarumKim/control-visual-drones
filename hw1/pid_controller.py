import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from computer_vision import ObjectDistanceAndPosition

class QuanserAero2Controller:
    def __init__(self, detectA, detectB):
        self.detectA = detectA
        self.detectB = detectB
        
        # Extraer ángulos iniciales y finales
        self.initial_angles = self.calculate_angles(detectA)
        self.final_angles = self.calculate_angles(detectB)

        # Parámetros del controlador PID
        self.kp = 5.0  # Aumentado para respuesta más rápida
        self.ki = 0.0  # Para eliminar error en estado estable
        self.kd = 0.0  # Para reducir oscilaciones
        
        # Variables de simulación
        self.dt = 0.05
        self.simulation_time = 10.0
        self.time_steps = int(self.simulation_time / self.dt)
        
        # Arrays para almacenar datos
        self.time_array = np.linspace(0, self.simulation_time, self.time_steps)
        self.angle_history = np.zeros((self.time_steps, 2))
        self.error_history = np.zeros((self.time_steps, 2))
        self.control_signal_history = np.zeros((self.time_steps, 2))
        
        # Estado inicial del sistema
        self.current_angles = self.initial_angles.copy()
        self.previous_error = [0.0, 0.0]
        self.integral_error = [0.0, 0.0]
        self.angular_velocity = [0.0, 0.0]  # Velocidad angular inicial
        
        # Control de convergencia
        self.converged = False
        self.convergence_time = 0
        self.convergence_threshold = 0.5  # Error menor a 0.5 grados
        self.stability_time = 1.0  # Tiempo que debe mantenerse estable (segundos)
        self.stable_count = 0
        
        self.inertia = 0.05  
        self.damping = 0.8   

    def calculate_angles(self, detector):
        """Calcula los ángulos de ambos rotores respecto a la referencia"""
        if not detector.centroids or not detector.reference_points:
            return [0.0, 0.0]
        
        px, py = detector.reference_points[0]
        centroid_data = []
        
        for (cx, cy) in detector.centroids:
            dx = cx - px
            dy = cy - py
            angle_rad = math.atan2(dy, dx)  # ángulo real (-180° a 180°)
            angle_deg = math.degrees(angle_rad)
            
            # Aplicar tu convención: si está en 3er o 4to cuadrante, poner negativo
            if angle_deg > 90:
                angle_deg -= 180  # convierte 91°..180° en -89°..0°
            elif angle_deg < -90:
                angle_deg += 180  # convierte -180°..-91° en 0°..-89°
                
            centroid_data.append((angle_deg, cx))
        
        centroid_data.sort(key=lambda x: x[1])  # Menor X → rotor izquierdo, mayor X → rotor derecho
        angles = [data[0] for data in centroid_data]
        while len(angles) < 2:
            angles.append(0.0)
        
        return angles[:2]

    def pid_controller(self, error, rotor_idx):
        """Implementa el controlador PID"""
        # Término proporcional
        proportional = self.kp * error
        
        # Término integral (con saturación para evitar windup)
        self.integral_error[rotor_idx] += error * self.dt
        self.integral_error[rotor_idx] = np.clip(self.integral_error[rotor_idx], -50, 50)
        integral = self.ki * self.integral_error[rotor_idx]
        
        # Término derivativo
        derivative_error = (error - self.previous_error[rotor_idx]) / self.dt
        derivative = self.kd * derivative_error
        
        # Señal de control total
        control_signal = proportional + integral + derivative
        
        # Saturación de la señal de control (limitación física del actuador)
        control_signal = np.clip(control_signal, -100, 100)
        
        # Actualizar error anterior
        self.previous_error[rotor_idx] = error
        
        
        return control_signal

    def system_dynamics(self, control_signal, rotor_idx):
        """Modela la dinámica del sistema Quanser Aero 2"""
        # Ecuación de movimiento simplificada: J*θ̈ + b*θ̇ = u
        # donde J es inercia, b es amortiguamiento, u es señal de control
        angular_acceleration = (control_signal - self.damping * self.angular_velocity[rotor_idx]) / self.inertia
        
        # Integración numérica (Euler)
        self.angular_velocity[rotor_idx] += angular_acceleration * self.dt
        self.current_angles[rotor_idx] += self.angular_velocity[rotor_idx] * self.dt
        
        noise = np.random.normal(0, 0.1)  # Ruido gaussiano pequeño
        self.current_angles[rotor_idx] += noise

    def run_simulation(self):
        """Simulación completa del sistema con PID"""
        print("Iniciando simulación...")
        print(f"Ángulos iniciales: {self.initial_angles}")
        print(f"Ángulos objetivo: {self.final_angles}")
        
        for i in range(self.time_steps):
            for rotor_idx in range(2):
                # Calcular error
                error = self.final_angles[rotor_idx] - self.current_angles[rotor_idx]
                
                # Controlador PID
                control_signal = self.pid_controller(error, rotor_idx)
                
                # Aplicar dinámica del sistema
                self.system_dynamics(control_signal, rotor_idx)
                
                # Guardar datos para gráficas
                self.error_history[i, rotor_idx] = error
                self.angle_history[i, rotor_idx] = self.current_angles[rotor_idx]
                self.control_signal_history[i, rotor_idx] = control_signal
            
            # Verificar convergencia
            max_error = max(abs(self.error_history[i, 0]), abs(self.error_history[i, 1]))
            if max_error < self.convergence_threshold:
                self.stable_count += 1
                if self.stable_count >= int(self.stability_time / self.dt) and not self.converged:
                    self.converged = True
                    self.convergence_time = self.time_array[i]
                    print(f"✅ Sistema convergido en t = {self.convergence_time:.2f}s")
                    print(f"Error final: Izq={self.error_history[i, 0]:.3f}°, Der={self.error_history[i, 1]:.3f}°")
                    
                    self.angle_history[i:, 0] = self.final_angles[0]
                    self.angle_history[i:, 1] = self.final_angles[1]
                    self.error_history[i:, 0] = 0
                    self.error_history[i:, 1] = 0
                    self.control_signal_history[i:, :] = 0
                    
                    # Truncar los arrays en el punto de convergencia
                    self.actual_steps = i + 1
                    break
            else:
                self.stable_count = 0
        
        if not self.converged:
            self.actual_steps = self.time_steps
            print("⚠️ Sistema no convergió completamente en el tiempo de simulación")
        
        print("Simulación completada.")
        print(f"Ángulos finales: [{self.current_angles[0]:.2f}°, {self.current_angles[1]:.2f}°]")


    def create_visualization(self):
        """Mostrar todas las ventanas de visualización"""
        self.create_images_window()
        self.create_system_diagram_window()
        self.create_angle_plots_window()
        self.create_control_analysis_window()
        self.create_animation_window()  # Nueva ventana de animación
        plt.show()

    def create_images_window(self):
        """Ventana 1: Imágenes de posiciones inicial y objetivo"""
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig1.suptitle('Posiciones del Quanser Aero 2', fontsize=16, fontweight='bold')
        
        imgA_positions = self.detectA.find_positions(box_thickness=5)
        imgB_positions = self.detectB.find_positions(box_thickness=5)
        
        ax1.imshow(imgA_positions)
        ax1.set_title("Posición Inicial (A)", fontsize=14, fontweight='bold', color='blue')
        ax1.axis('off')
        
        ax2.imshow(imgB_positions)
        ax2.set_title("Posición Objetivo (B)", fontsize=14, fontweight='bold', color='red')
        ax2.axis('off')
        
        info_text = f"Ángulos Iniciales:\nIzq: {self.initial_angles[0]:.1f}°\nDer: {-self.initial_angles[1]:.1f}°"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        info_text2 = f"Ángulos Objetivo:\nIzq: {self.final_angles[0]:.1f}°\nDer: {-self.final_angles[1]:.1f}°"
        ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()

    def create_animation_window(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Posición Inicial y Final - Quanser Aero 2', fontsize=16, fontweight='bold')

        def draw_bar(ax, angles, title, color):
            bar_length = 1.5
            rotor_distance = bar_length / 2
            angle_left, angle_right = angles
            
            # Coordenadas relativas de los rotores desde el pivote (0,0)
            left_x = -rotor_distance * np.cos(np.radians(angle_left))
            left_y = -rotor_distance * np.sin(np.radians(angle_left))
            right_x = rotor_distance * np.cos(np.radians(angle_right))
            right_y = rotor_distance * np.sin(np.radians(angle_right))
            
            # Barra
            ax.plot([left_x, right_x], [left_y, right_y], color=color, linewidth=8)
            
            # Rotores
            ax.plot(left_x, left_y, 'ro', markersize=14, label="Rotor Izq")
            ax.plot(right_x, right_y, 'go', markersize=14, label="Rotor Der")
            
            # Pivote
            ax.plot(0, 0, 'ko', markersize=10)

            # Base (tronco) centrada en el pivote
            base_width = 0.3
            base_height = 2.0
            tronco = plt.Rectangle((-base_width/2, -base_height), 
                                base_width, base_height,
                                color="dimgray", alpha=0.9)
            ax.add_patch(tronco)

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-1.8, 1.8)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Ahora sí: inicial y final correctamente
        draw_bar(ax1, self.initial_angles, "Posición Inicial", "blue")
        draw_bar(ax2, self.final_angles, "Posición Final (alcanzada)", "red")

        plt.tight_layout()


    def create_system_diagram_window(self):
        """Ventana 2: Diagrama del sistema de control"""
        fig5, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig5.suptitle('Diagrama de Bloques del Sistema de Control', fontsize=16, fontweight='bold')
        self.draw_system_diagram(ax)
        plt.tight_layout()

    def draw_system_diagram(self, ax):
        """Dibuja el diagrama de control"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
        
        # Bloques
        pid_rect = plt.Rectangle((1, 2), 1.5, 1, fill=True, facecolor='lightblue', 
                                edgecolor='navy', linewidth=2)
        ax.add_patch(pid_rect)
        ax.text(1.75, 2.5, 'PID\nController', ha='center', va='center', 
                fontweight='bold', fontsize=10)
        
        system_rect = plt.Rectangle((4, 2), 2, 1, fill=True, facecolor='lightgreen', 
                                   edgecolor='darkgreen', linewidth=2)
        ax.add_patch(system_rect)
        ax.text(5, 2.5, 'Quanser\nAero 2', ha='center', va='center', 
                fontweight='bold', fontsize=10)
        
        sensor_rect = plt.Rectangle((7, 2), 1.5, 1, fill=True, facecolor='lightyellow', 
                                   edgecolor='orange', linewidth=2)
        ax.add_patch(sensor_rect)
        ax.text(7.75, 2.5, 'Visión\nComputacional', ha='center', va='center', 
                fontweight='bold', fontsize=9)
        
        # Flechas y etiquetas (mismo código que antes)
        ax.arrow(0.5, 2.5, 0.4, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
        ax.arrow(2.6, 2.5, 1.2, 0, head_width=0.15, head_length=0.1, fc='red', ec='red', linewidth=2)
        ax.arrow(6.1, 2.5, 0.7, 0, head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
        ax.arrow(7.75, 1.9, 0, -0.7, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
        ax.arrow(7.6, 1, -6.8, 0, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
        ax.arrow(0.8, 1, 0, 1.3, head_width=0.15, head_length=0.1, fc='purple', ec='purple', linewidth=2)
        
        # Etiquetas
        ax.text(0.2, 2.8, 'Referencia\n(θ_ref)', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.text(3.2, 2.8, 'Señal Control\n(u)', ha='center', va='bottom', fontweight='bold', fontsize=9, color='red')
        ax.text(6.6, 2.8, 'Ángulo\n(θ)', ha='center', va='bottom', fontweight='bold', fontsize=9, color='blue')
        ax.text(4, 0.7, 'Retroalimentación', ha='center', va='center', fontweight='bold', fontsize=10, color='green')
        ax.text(0.3, 1.5, 'Error\n(e)', ha='center', va='center', rotation=90, fontweight='bold', fontsize=9, color='purple')
        
        # Sumador
        circle = Circle((0.8, 2.5), 0.2, fill=True, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.72, 2.5, '+', ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.88, 2.35, '−', ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Parámetros PID
        params_text = f"Parámetros PID:\nKp = {self.kp}\nKi = {self.ki}\nKd = {self.kd}"
        ax.text(1.75, 4, params_text, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', alpha=0.8),
                fontsize=9, fontweight='bold')
        
        ax.set_title('Sistema de Control en Lazo Cerrado', fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')

    def create_angle_plots_window(self):
        """Ventana 3: Evolución de los ángulos de los rotores"""
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Evolución de los Ángulos de los Rotores', fontsize=16, fontweight='bold')
        
        # Rotor 1
        ax[0].plot(self.time_array, self.angle_history[:, 0], label="Ángulo Rotor 1", color="blue", linewidth=2)
        ax[0].hlines(self.final_angles[0], self.time_array[0], self.time_array[-1], colors="red", linestyles="--", label="Referencia", linewidth=2)
        ax[0].set_ylabel("Ángulo (°)", fontsize=12)
        ax[0].set_title("Rotor 1 (Izquierdo)", fontsize=14)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Rotor 2
        ax[1].plot(self.time_array, self.angle_history[:, 1], label="Ángulo Rotor 2", color="green", linewidth=2)
        ax[1].hlines(self.final_angles[1], self.time_array[0], self.time_array[-1], colors="red", linestyles="--", label="Referencia", linewidth=2)
        ax[1].set_xlabel("Tiempo (s)", fontsize=12)
        ax[1].set_ylabel("Ángulo (°)", fontsize=12)
        ax[1].set_title("Rotor 2 (Derecho)", fontsize=14)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()

    def create_control_analysis_window(self):
        """Ventana 4: Análisis de error y señales de control"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis del Sistema de Control', fontsize=16, fontweight='bold')
        
        # Error vs tiempo - Rotor 1
        ax1.plot(self.time_array, self.error_history[:, 0], 'b-', linewidth=2, label='Error Rotor 1')
        ax1.set_ylabel('Error (°)', fontsize=11)
        ax1.set_title('Error de Seguimiento - Rotor 1', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Error vs tiempo - Rotor 2
        ax2.plot(self.time_array, self.error_history[:, 1], 'g-', linewidth=2, label='Error Rotor 2')
        ax2.set_ylabel('Error (°)', fontsize=11)
        ax2.set_title('Error de Seguimiento - Rotor 2', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Señal de control - Rotor 1
        ax3.plot(self.time_array, self.control_signal_history[:, 0], 'r-', linewidth=2, label='Control Rotor 1')
        ax3.set_xlabel('Tiempo (s)', fontsize=11)
        ax3.set_ylabel('Señal Control', fontsize=11)
        ax3.set_title('Señal de Control - Rotor 1', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Señal de control - Rotor 2
        ax4.plot(self.time_array, self.control_signal_history[:, 1], 'm-', linewidth=2, label='Control Rotor 2')
        ax4.set_xlabel('Tiempo (s)', fontsize=11)
        ax4.set_ylabel('Señal Control', fontsize=11)
        ax4.set_title('Señal de Control - Rotor 2', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()

def main():
    detectA = ObjectDistanceAndPosition("posA.jpeg")
    detectB = ObjectDistanceAndPosition("posB.jpeg")
    
    detectA.reference_points = [(782, 420)]
    detectB.reference_points = [(780, 426)]
    
    detectA.find_positions(box_thickness=5)
    detectB.find_positions(box_thickness=5)
    
    controller = QuanserAero2Controller(detectA, detectB)
    controller.run_simulation()
    controller.create_visualization()


if __name__ == "__main__":
    main()