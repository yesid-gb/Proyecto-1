from abc import ABC, abstractmethod  # Importa clases para crear clases abstractas
import math                 # Importa NumPy para cálculos matemáticos y vectoriales
import matplotlib.pyplot as plt      # Importa Matplotlib para visualización gráfica
import numpy as np

# Clase para cuerpos celestes - Clase base abstracta para todos los cuerpos celestes
class CuerpoCeleste(ABC):
    def __init__(self, nombre, masa, radio=1, x=0, y=0, z=0, vx=0, vy=0, vz=0):  # Constructor con valores predeterminados
        self._nombre = nombre        # Nombre del cuerpo celeste
        self._masa = masa            # Masa del cuerpo celeste
        self._radio = radio          # Radio físico del cuerpo
        self._x = x                  # Posición x inicial
        self._y = y                  # Posición y inicial
        self._z = z                  # Posición z inicial
        self._vx = vx                # Velocidad en dirección x inicial
        self._vy = vy                # Velocidad en dirección y inicial
        self._vz = vz                # Velocidad en dirección z inicial
        self._historial_posiciones = []  # Lista para almacenar el historial de posiciones
        self._ax = 0                 # Aceleración en dirección x (para el algoritmo Verlet)
        self._ay = 0                 # Aceleración en dirección y (para el algoritmo Verlet)
        self._az = 0                 # Aceleración en dirección z (para el algoritmo Verlet)

    @property
    def nombre(self):                # Getter para el nombre
        return self._nombre
    
    @property
    def masa(self):                  # Getter para la masa
        return self._masa
    
    @property
    def radio(self):                 # Getter para el radio
        return self._radio
    
    @property
    def posicion(self):              # Getter que devuelve posición como vector 
        return [self._x, self._y, self._z]
    
    @property
    def velocidad(self):             # Getter que devuelve velocidad como vector 
        return [self._vx, self._vy, self._vz]
    
    @property
    def historial_posiciones(self):  # Getter para acceder al historial de posiciones
        return self._historial_posiciones
    
    def agregar_posicion_historial(self):  # Método para guardar la posición actual en el historial
        self._historial_posiciones.append([self._x, self._y, self._z])
    
    @abstractmethod
    def actualizar_posicion(self, dt, vector_traslacion=None):  # Método abstracto que deben implementar las subclases
        pass

# Clase para planetas - Hereda de CuerpoCeleste
class Planeta(CuerpoCeleste):
    def actualizar_posicion(self, dt, vector_traslacion=None):  # Implementación del método abstracto
        # Guardar posición anterior
        x_prev, y_prev, z_prev = self._x, self._y, self._z  # Guarda posición previa para cálculos
        
        # Actualizar posición con Verlet (algoritmo de integración numérica de segundo orden)
        self._x += self._vx * dt + 0.5 * self._ax * dt**2  # Actualiza posición x usando integración Verlet
        self._y += self._vy * dt + 0.5 * self._ay * dt**2  # Actualiza posición y usando integración Verlet
        self._z += self._vz * dt + 0.5 * self._az * dt**2  # Actualiza posición z usando integración Verlet
        
        # Aplicar vector de traslación global si existe
        if vector_traslacion is not None:  # Comprueba si hay vector de traslación
            self._x += vector_traslacion[0] * dt  # Aplica traslación en x
            self._y += vector_traslacion[1] * dt  # Aplica traslación en y
            self._z += vector_traslacion[2] * dt  # Aplica traslación en z
        
        # Guardar aceleración anterior
        ax_prev, ay_prev, az_prev = self._ax, self._ay, self._az  # Guarda aceleración previa para cálculos Verlet
        
        # Actualizar velocidad con Verlet
        self._vx += 0.5 * (ax_prev + self._ax) * dt  # Actualiza velocidad x usando el promedio de aceleración
        self._vy += 0.5 * (ay_prev + self._ay) * dt  # Actualiza velocidad y usando el promedio de aceleración
        self._vz += 0.5 * (az_prev + self._az) * dt  # Actualiza velocidad z usando el promedio de aceleración
        
        self.agregar_posicion_historial()  # Agrega la nueva posición al historial

# Clase para estrellas - Hereda de CuerpoCeleste
class Estrella(CuerpoCeleste):
    def actualizar_posicion(self, dt, vector_traslacion=None):  # Implementación para estrellas (solo traslación)
        # Aplicar vector de traslación global si existe
        if vector_traslacion is not None:  # Comprueba si hay vector de traslación
            self._x += vector_traslacion[0] * dt  # Aplica traslación en x
            self._y += vector_traslacion[1] * dt  # Aplica traslación en y
            self._z += vector_traslacion[2] * dt  # Aplica traslación en z
            
        self.agregar_posicion_historial()  # Agrega la nueva posición al historial


G = 6.67430e-11  # Constante universal de gravitación en m³/(kg·s²)

def calcular_fuerza(c1, c2):  # Función para calcular la fuerza gravitacional entre dos cuerpos
    rx = c2.posicion[0] - c1.posicion[0]  # Componente x del vector distancia
    ry = c2.posicion[1] - c1.posicion[1]  # Componente y del vector distancia
    rz = c2.posicion[2] - c1.posicion[2]  # Componente z del vector distancia
    distancia = math.sqrt(rx*rx + ry*ry + rz*rz)  # Magnitud de la distancia

    if distancia == 0:  # Evita división por cero
        return [0, 0, 0]  # Devuelve vector de fuerza cero
    
    # Calcular fuerza gravitacional: F = G * m1 * m2 / r^2
    fuerza_magnitud = G * c1.masa * c2.masa / (distancia**2)  # Magnitud de la fuerza según ley de gravitación
    fx = fuerza_magnitud * rx / distancia  # Componente x del vector fuerza
    fy = fuerza_magnitud * ry / distancia  # Componente y del vector fuerza
    fz = fuerza_magnitud * rz / distancia  # Componente z del vector fuerza
    fuerza_vector = [fx, fy, fz]  # Vector fuerza completo  
    
    return fuerza_vector  # Devuelve el vector de fuerza gravitacional

def simular_sistema(cuerpos, dt, pasos, vector_traslacion=None):  # Función principal para simular el sistema
    """
    Simula el sistema con un vector de traslación opcional
    
    Args:
        cuerpos: Lista de cuerpos celestes
        dt: Paso de tiempo
        pasos: Número de pasos de simulación
        vector_traslacion: Vector [vx, vy, vz] de velocidad global del sistema
    """
    # Inicializar historial de posiciones
    for cuerpo in cuerpos:  # Recorre todos los cuerpos
        cuerpo.agregar_posicion_historial()  # Agrega posición inicial al historial
    
    # Paso inicial: calcular aceleraciones iniciales para todos los cuerpos
    for cuerpo in cuerpos:  # Recorre todos los cuerpos
        fuerza_total = [0, 0, 0]  # Inicializa vector de fuerza total
        for otro in cuerpos:  # Recorre todos los otros cuerpos
            if otro != cuerpo:  # Evita calcular fuerza con el mismo cuerpo
                fuerza_total += calcular_fuerza(cuerpo, otro)  # Suma la fuerza ejercida por el otro cuerpo
        
        # Calcular aceleración según la ley de Newton: a = F/m
        cuerpo._ax = fuerza_total[0] / cuerpo.masa  # Calcula componente x de aceleración
        cuerpo._ay = fuerza_total[1] / cuerpo.masa  # Calcula componente y de aceleración
        cuerpo._az = fuerza_total[2] / cuerpo.masa  # Calcula componente z de aceleración
    
    for _ in range(pasos):  # Bucle principal de simulación
        # Actualizar posiciones
        for cuerpo in cuerpos:  # Recorre todos los cuerpos
            cuerpo.actualizar_posicion(dt, vector_traslacion)  # Actualiza posición de cada cuerpo
        
        # Calcular nuevas aceleraciones para todos los cuerpos
        for cuerpo in cuerpos:  # Recorre todos los cuerpos
            fuerza_total = [0, 0, 0]  # vector de fuerza total
            for otro in cuerpos:  # Recorre todos los otros cuerpos
                if otro != cuerpo:  # Evita calcular fuerza con el mismo cuerpo
                    f = calcular_fuerza(cuerpo, otro)  # Calcula la fuerza ejercida por el otro cuerpo
                    fuerza_total[0] += f[0]  # Suma componente x
                    fuerza_total[1] += f[1]  # Suma componente y
                    fuerza_total[2] += f[2]  # Suma componente z  
            
            # Calcular aceleración según la ley de Newton: a = F/m
            cuerpo._ax = fuerza_total[0] / cuerpo.masa  # Actualiza componente x de aceleración
            cuerpo._ay = fuerza_total[1] / cuerpo.masa  # Actualiza componente y de aceleración
            cuerpo._az = fuerza_total[2] / cuerpo.masa  # Actualiza componente z de aceleración

def visualizar_sistema_3d(cuerpos, pasos_animacion=None):  # Función para visualizar la simulación en 3D
    """
    Visualiza el sistema en 3D con rotación interactiva
    
    Args:
        cuerpos: Lista de cuerpos celestes
        pasos_animacion: Número de pasos de animación
    """
    # Configurar figura con interactividad
    plt.ion()  # Activar modo interactivo para manipulación en tiempo real
    fig = plt.figure(figsize=(12, 10))  # Crear figura con tamaño específico
    ax = fig.add_subplot(111, projection='3d')  # Agregar subplot 3D
    
    # Configurar aspecto 3D para mantener escala igual en todos los ejes
    ax.set_box_aspect([1, 1, 1])  # Establece relación de aspecto igual en todos los ejes
    
    ax.set_xlabel('X (m)')  # Etiqueta para eje X
    ax.set_ylabel('Y (m)')  # Etiqueta para eje Y
    ax.set_zlabel('Z (m)')  # Etiqueta para eje Z
    ax.set_title('Simulación de Órbitas en 3D Interactiva')  # Título de la figura
    
    # Elementos de visualización con esferas 3D
    elementos = {}  # Diccionario para almacenar elementos de visualización
    for i, cuerpo in enumerate(cuerpos):  # Recorre todos los cuerpos
        # Asignar colores específicos por nombre
        if isinstance(cuerpo, Estrella):  # Si es una estrella
            color = 'gold'  # Color dorado para estrellas
            radio_vis = np.log10(cuerpo.masa) * 0.1  # Radio visual basado en logaritmo de la masa
        elif cuerpo.nombre == "Tierra":  # Si es la Tierra
            color = 'blue'  # Color azul para la Tierra
            radio_vis = 0.5  # Radio visual relativo para la Tierra
        elif cuerpo.nombre == "Marte":  # Si es Marte
            color = 'red'  # Color rojo para Marte
            radio_vis = 0.3  # Radio visual relativo para Marte
        else:  # Para otros cuerpos
            color = plt.cm.tab10(i % 10)  # Color de la paleta tab10
            radio_vis = 1.0  # Radio visual predeterminado
            
        elementos[cuerpo.nombre] = {  # Almacena elementos de visualización para cada cuerpo
            'esfera': None,  # Placeholder para la esfera (se inicializará en cada frame)
            'linea': ax.plot([], [], [], '-', color=color, alpha=0.7)[0],  # Línea para la trayectoria
            'color': color,  # Color asignado
            'radio': radio_vis  # Radio visual
        }
    
    # Configurar leyenda personalizada
    from matplotlib.lines import Line2D  # Importa Line2D para crear elementos de leyenda personalizados
    leyenda_elementos = [Line2D([0], [0], marker='o', color='w', markerfacecolor=elementos[cuerpo.nombre]['color'],
                               markersize=10, label=cuerpo.nombre) for cuerpo in cuerpos]  # Crea elementos de leyenda
    ax.legend(handles=leyenda_elementos, loc='upper right')  # Añade leyenda en esquina superior derecha
    
    plt.tight_layout()  # Ajusta automáticamente los elementos para optimizar espacio
    
    # Configurar animación manual
    if pasos_animacion is None:  # Si no se especifica número de pasos
        pasos_animacion = len(cuerpos[0].historial_posiciones)  # Usa todos los pasos disponibles
    
    pasos_animacion = min(pasos_animacion, len(cuerpos[0].historial_posiciones))  # Limita a los pasos disponibles
    
    print("Animación iniciada. Puedes rotar la figura con el mouse mientras se actualiza.")  # Mensaje informativo
    print(f"Mostrando {pasos_animacion} pasos de simulación.")  # Mensaje informativo
    
    # Función para crear esferas 3D
    def crear_esfera(centro, radio, color):  # Función que crea una esfera 3D
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)# Crear matrices para las coordenadas de la esfera
        x = centro[0] + radio * np.outer(np.cos(u), np.sin(v))
        y = centro[1] + radio * np.outer(np.sin(u), np.sin(v))
        z = centro[2] + radio * np.outer(np.ones(np.size(u)), np.cos(v))
        return ax.plot_surface(x, y, z, color=color, alpha=1)  # Devuelve superficie 3D
    
    # Actualizar manualmente frame por frame para permitir interactividad
    for frame in range(pasos_animacion):  # Bucle para cada fotograma
        # Limpiar esferas previas
        for cuerpo in cuerpos:  # Recorre todos los cuerpos
            if elementos[cuerpo.nombre]['esfera'] is not None:  # Si existe una esfera previa
                elementos[cuerpo.nombre]['esfera'].remove()  # Elimina la esfera anterior
        
        # Calcular el centro del sistema para este frame
        posiciones_actuales = [cuerpo.historial_posiciones[frame] for cuerpo in cuerpos]  # Lista de posiciones actuales
        centro_x = sum(pos[0] for pos in posiciones_actuales) / len(posiciones_actuales)
        centro_y = sum(pos[1] for pos in posiciones_actuales) / len(posiciones_actuales)
        centro_z = sum(pos[2] for pos in posiciones_actuales) / len(posiciones_actuales)
        centro_sistema = [centro_x, centro_y, centro_z]  # Centro del sistema como promedio de posiciones
        
        # Establecer nuevos límites para seguir el movimiento del sistema
        # Usar escala dinámica para cubrir todos los cuerpos con margen
        max_dist = 0
        for pos in posiciones_actuales:
            dx = pos[0] - centro_sistema[0]
            dy = pos[1] - centro_sistema[1]
            dz = pos[2] - centro_sistema[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            if dist > max_dist:
                max_dist = dist
        margen = max_dist * 1.5  # Factor de margen para la vista
        
        # Actualizar límites de los ejes para seguir el centro del sistema
        ax.set_xlim(centro_sistema[0] - margen, centro_sistema[0] + margen)  # Actualiza límites eje x
        ax.set_ylim(centro_sistema[1] - margen, centro_sistema[1] + margen)  # Actualiza límites eje y
        ax.set_zlim(centro_sistema[2] - margen, centro_sistema[2] + margen)  # Actualiza límites eje z
        
        # Factor de escala para hacer visibles los cuerpos pero manteniendo el tamaño proporcional
        escala = margen * 0.05  # Calcula factor de escala para visualización
        
        # Actualizar planetas y trayectorias
        for cuerpo in cuerpos:  # Recorre todos los cuerpos
            historial = cuerpo.historial_posiciones[:frame+10]  # Obtiene historial hasta frame actual +10
            if not historial:  # Si no hay historial
                continue  # Salta a la siguiente iteración
                
            pos_actual = historial[-1]  # Posición actual del cuerpo
            
            # Crear esfera 3D para representar el cuerpo
            radio_escalado = elementos[cuerpo.nombre]['radio'] * escala  # Escala el radio visual
            elementos[cuerpo.nombre]['esfera'] = crear_esfera(
                pos_actual, radio_escalado, elementos[cuerpo.nombre]['color']  # Crea esfera en la posición actual
            )
            
            # Actualizar trayectoria
            x, y, z = zip(*historial)  # Desempaqueta coordenadas para la trayectoria
            elementos[cuerpo.nombre]['linea'].set_data(x, y)  # Actualiza datos x,y de la línea
            elementos[cuerpo.nombre]['linea'].set_3d_properties(z)  # Actualiza datos z de la línea
        
        # Mostrar progreso
        if frame % 10 == 0:  # Cada 10 frames
            print(f"Progreso: {frame}/{pasos_animacion}", end="\r")  # Muestra progreso de la animación
        
        # Actualizar canvas y pausar para permitir interactividad
        fig.canvas.draw()  # Redibuja el canvas
        plt.pause(0.0001)  # Pausa breve para permitir interactividad y ajustar velocidad de animación
    
    print("\nAnimación completada. La figura permanece interactiva.")  # Mensaje informativo
    plt.ioff()  # Desactivar modo interactivo cuando termina la animación
    plt.show()  # Mantener la ventana abierta para interacción


if __name__ == "__main__":  # Bloque de ejecución principal
    # Sistema solar simplificado con radios reales (en metros)
    # Los radios son usados solo para visualización y están escalados
    sol = Estrella("Sol", 1.989e30, radio=6.957e8)  # Crea objeto Sol (estrella)
    
    # Tierra orbita
    tierra = Planeta(
        "Tierra", 5.972e24, radio=6.371e6,  # Crea objeto Tierra (planeta)
        x=1.496e11, y=0, z=0,  # Posición inicial (distancia al Sol)
        vx=0, vy=29.78e3, vz=1.0e3  # Velocidad inicial (órbita)
    )
    
    # Marte orbita
    marte = Planeta(
        "Marte", 6.39e23, radio=3.389e6,  # Crea objeto Marte (planeta)
        x=2.279e11, y=0, z=0,  # Posición inicial (distancia al Sol)
        vx=0, vy=23.0e3, vz=2.0e3  # Velocidad inicial (órbita)
    )
    
    # Vector de traslación global [vx, vy, vz] en m/s
    vector_traslacion = [5.0e3, 8.0e3, 15.0e3]  # 5 km/s en x, 8 km/s en y, 15 km/s en z
    
    # Simulación con paso de tiempo (1 hora) y vector de traslación
    print("Iniciando simulación...")  
    simular_sistema([sol, tierra, marte], dt=360000, pasos=365*24, vector_traslacion=vector_traslacion)  # Ejecuta simulación
    print("Simulación completada. Iniciando visualización...")  
    
    # Visualización interactiva
    visualizar_sistema_3d([sol, tierra, marte], pasos_animacion=10000)  # Muestra visualización 3D