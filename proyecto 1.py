from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


# Clase para cuerpos celestes
class CuerpoCeleste(ABC):
    def __init__(self, nombre, masa, radio=1, x=0, y=0, z=0, vx=0, vy=0, vz=0):
        self._nombre = nombre
        self._masa = masa
        self._radio = radio  
        self._x = x
        self._y = y
        self._z = z
        self._vx = vx
        self._vy = vy
        self._vz = vz
        self._historial_posiciones = []
        self._ax = 0  # Aceleraciones para Verlet
        self._ay = 0
        self._az = 0

    @property
    def nombre(self):
        return self._nombre
    
    @property
    def masa(self):
        return self._masa
    
    @property
    def radio(self):
        return self._radio
    
    @property
    def posicion(self):
        return np.array([self._x, self._y, self._z])
    
    @property
    def velocidad(self):
        return np.array([self._vx, self._vy, self._vz])
    
    @property
    def historial_posiciones(self):
        return self._historial_posiciones
    
    def agregar_posicion_historial(self):
        self._historial_posiciones.append(self.posicion.copy())
    
    @abstractmethod
    def actualizar_posicion(self, dt, vector_traslacion=None):
        pass

# Clase para planetas
class Planeta(CuerpoCeleste):
    def actualizar_posicion(self, dt, vector_traslacion=None):
        # Guardar posición anterior
        x_prev, y_prev, z_prev = self._x, self._y, self._z
        
        # Actualizar posición con Verlet
        self._x += self._vx * dt + 0.5 * self._ax * dt**2
        self._y += self._vy * dt + 0.5 * self._ay * dt**2
        self._z += self._vz * dt + 0.5 * self._az * dt**2
        
        # Aplicar vector de traslación global si existe
        if vector_traslacion is not None:
            self._x += vector_traslacion[0] * dt
            self._y += vector_traslacion[1] * dt
            self._z += vector_traslacion[2] * dt
        
        # Guardar aceleración anterior
        ax_prev, ay_prev, az_prev = self._ax, self._ay, self._az
        
        # Actualizar velocidad con Verlet
        self._vx += 0.5 * (ax_prev + self._ax) * dt
        self._vy += 0.5 * (ay_prev + self._ay) * dt
        self._vz += 0.5 * (az_prev + self._az) * dt
        
        self.agregar_posicion_historial()

# Clase para estrellas
class Estrella(CuerpoCeleste):
    def actualizar_posicion(self, dt, vector_traslacion=None):
        # Aplicar vector de traslación global si existe
        if vector_traslacion is not None:
            self._x += vector_traslacion[0] * dt
            self._y += vector_traslacion[1] * dt
            self._z += vector_traslacion[2] * dt
            
        self.agregar_posicion_historial()

# Constante gravitacional
G = 6.67430e-11

def calcular_fuerza(c1, c2):
    r = c2.posicion - c1.posicion
    distancia = np.linalg.norm(r)
    
    if distancia == 0:
        return np.zeros(3)
    
    # Calcular fuerza gravitacional: F = G * m1 * m2 / r^2
    fuerza_magnitud = G * c1.masa * c2.masa / (distancia**2)
    fuerza_vector = fuerza_magnitud * r / distancia
    
    return fuerza_vector

def simular_sistema(cuerpos, dt, pasos, vector_traslacion=None):
    """
    Simula el sistema con un vector de traslación opcional
    
    Args:
        cuerpos: Lista de cuerpos celestes
        dt: Paso de tiempo
        pasos: Número de pasos de simulación
        vector_traslacion: Vector [vx, vy, vz] de velocidad global del sistema
    """
    # Inicializar historial de posiciones
    for cuerpo in cuerpos:
        cuerpo.agregar_posicion_historial()
    
    # Paso inicial: calcular aceleraciones iniciales para todos los cuerpos
    for cuerpo in cuerpos:
        fuerza_total = np.zeros(3)
        for otro in cuerpos:
            if otro != cuerpo:
                fuerza_total += calcular_fuerza(cuerpo, otro)
        
        # Calcular aceleración según la ley de Newton: a = F/m
        cuerpo._ax = fuerza_total[0] / cuerpo.masa
        cuerpo._ay = fuerza_total[1] / cuerpo.masa
        cuerpo._az = fuerza_total[2] / cuerpo.masa
    
    for _ in range(pasos):
        # Actualizar posiciones
        for cuerpo in cuerpos:
            cuerpo.actualizar_posicion(dt, vector_traslacion)
        
        # Calcular nuevas aceleraciones para todos los cuerpos
        for cuerpo in cuerpos:
            fuerza_total = np.zeros(3)
            for otro in cuerpos:
                if otro != cuerpo:
                    fuerza_total += calcular_fuerza(cuerpo, otro)
            
            # Calcular aceleración según la ley de Newton: a = F/m
            cuerpo._ax = fuerza_total[0] / cuerpo.masa
            cuerpo._ay = fuerza_total[1] / cuerpo.masa
            cuerpo._az = fuerza_total[2] / cuerpo.masa

def visualizar_sistema_3d(cuerpos, pasos_animacion=None):
    """
    Visualiza el sistema en 3D con rotación interactiva
    
    Args:
        cuerpos: Lista de cuerpos celestes
        pasos_animacion: Número de pasos de animación
    """
    # Configurar figura con interactividad
    plt.ion()  # Activar modo interactivo
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Configurar aspecto 3D para mantener escala igual en todos los ejes
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Simulación de Órbitas en 3D Interactiva')
    
    # Elementos de visualización con esferas 3D
    elementos = {}
    for i, cuerpo in enumerate(cuerpos):
        # Asignar colores específicos por nombre
        if isinstance(cuerpo, Estrella):
            color = 'gold'
            radio_vis = np.log10(cuerpo.masa) * 0.1
        elif cuerpo.nombre == "Tierra":
            color = 'blue'
            radio_vis = 0.5  # Radio visual relativo
        elif cuerpo.nombre == "Marte":
            color = 'red'
            radio_vis = 0.3  # Radio visual relativo
        else:
            color = plt.cm.tab10(i % 10)
            radio_vis = 1.0
            
        elementos[cuerpo.nombre] = {
            'esfera': None,  # Se inicializará en cada frame
            'linea': ax.plot([], [], [], '-', color=color, alpha=0.7)[0],
            'color': color,
            'radio': radio_vis
        }
    
    # Configurar leyenda personalizada
    from matplotlib.lines import Line2D
    leyenda_elementos = [Line2D([0], [0], marker='o', color='w', markerfacecolor=elementos[cuerpo.nombre]['color'],
                               markersize=10, label=cuerpo.nombre) for cuerpo in cuerpos]
    ax.legend(handles=leyenda_elementos, loc='upper right')
    
    plt.tight_layout()
    
    # Configurar animación manual
    if pasos_animacion is None:
        pasos_animacion = len(cuerpos[0].historial_posiciones)
    
    pasos_animacion = min(pasos_animacion, len(cuerpos[0].historial_posiciones))
    
    print("Animación iniciada. Puedes rotar la figura con el mouse mientras se actualiza.")
    print(f"Mostrando {pasos_animacion} pasos de simulación.")
    
    # Función para crear esferas 3D
    def crear_esfera(centro, radio, color):
        u = np.linspace(0, 2 * np.pi, 10)
        v = np.linspace(0, np.pi, 10)
        x = centro[0] + radio * np.outer(np.cos(u), np.sin(v))
        y = centro[1] + radio * np.outer(np.sin(u), np.sin(v))
        z = centro[2] + radio * np.outer(np.ones(np.size(u)), np.cos(v))
        return ax.plot_surface(x, y, z, color=color, alpha=1)
    
    # Actualizar manualmente frame por frame para permitir interactividad
    for frame in range(pasos_animacion):
        # Limpiar esferas previas
        for cuerpo in cuerpos:
            if elementos[cuerpo.nombre]['esfera'] is not None:
                elementos[cuerpo.nombre]['esfera'].remove()
        
        # Calcular el centro del sistema para este frame
        posiciones_actuales = [cuerpo.historial_posiciones[frame] for cuerpo in cuerpos]
        centro_sistema = np.mean(posiciones_actuales, axis=0)
        
        # Establecer nuevos límites para seguir el movimiento del sistema
        # Usar escala dinámica para cubrir todos los cuerpos con margen
        posiciones_np = np.array(posiciones_actuales)
        max_dist = np.max(np.linalg.norm(posiciones_np - centro_sistema, axis=1))
        margen = max_dist * 1.5  # Factor de margen para la vista
        
        # Actualizar límites de los ejes para seguir el centro del sistema
        ax.set_xlim(centro_sistema[0] - margen, centro_sistema[0] + margen)
        ax.set_ylim(centro_sistema[1] - margen, centro_sistema[1] + margen)
        ax.set_zlim(centro_sistema[2] - margen, centro_sistema[2] + margen)
        
        # Factor de escala para hacer visibles los cuerpos pero manteniendo el tamaño proporcional
        escala = margen * 0.05
        
        # Actualizar planetas y trayectorias
        for cuerpo in cuerpos:
            historial = cuerpo.historial_posiciones[:frame+10]
            if not historial:
                continue
                
            pos_actual = historial[-1]
            
            # Crear esfera 3D para representar el cuerpo
            radio_escalado = elementos[cuerpo.nombre]['radio'] * escala
            elementos[cuerpo.nombre]['esfera'] = crear_esfera(
                pos_actual, radio_escalado, elementos[cuerpo.nombre]['color']
            )
            
            # Actualizar trayectoria
            x, y, z = zip(*historial)
            elementos[cuerpo.nombre]['linea'].set_data(x, y)
            elementos[cuerpo.nombre]['linea'].set_3d_properties(z)
        
        # Mostrar progreso
        if frame % 10 == 0:
            print(f"Progreso: {frame}/{pasos_animacion}", end="\r")
        
        # Actualizar canvas y pausar para permitir interactividad
        fig.canvas.draw()
        plt.pause(0.0001)  # Ajusta este valor para cambiar la velocidad de la animación
    
    print("\nAnimación completada. La figura permanece interactiva.")
    plt.ioff()  # Desactivar modo interactivo cuando termina la animación
    plt.show()  # Mantener la ventana abierta para interacción


if __name__ == "__main__":
    # Sistema solar simplificado con radios reales (en metros)
    # Los radios son usados solo para visualización y están escalados
    sol = Estrella("Sol", 1.989e30, radio=6.957e8)
    
    # Tierra orbita
    tierra = Planeta(
        "Tierra", 5.972e24, radio=6.371e6,
        x=1.496e11, y=0, z=0,
        vx=0, vy=29.78e3, vz=1.0e3
    )
    
    # Marte orbita
    marte = Planeta(
        "Marte", 6.39e23, radio=3.389e6,
        x=2.279e11, y=0, z=0,
        vx=0, vy=23.0e3, vz=2.0e3
    )
    
    # Vector de traslación global [vx, vy, vz] en m/s
    vector_traslacion = np.array([5.0e3, 8.0e3, 15.0e3])  # 5 km/s en x, 8 km/s en y, 15 km/s en z
    
    # Simulación con paso de tiempo (1 hora) y vector de traslación
    print("Iniciando simulación...")
    simular_sistema([sol, tierra, marte], dt=360000, pasos=365*24, vector_traslacion=vector_traslacion)
    print("Simulación completada. Iniciando visualización...")
    
    # Visualización interactiva
    visualizar_sistema_3d([sol, tierra, marte], pasos_animacion=10000)  # Mostrar un año de simulación (365*24)