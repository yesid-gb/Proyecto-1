# Simulador de Sistema Solar 3D

Este proyecto implementa un simulador de órbitas planetarias en 3D con visualización interactiva, utilizando las leyes de gravitación universal de Newton y el algoritmo de integración de Verlet para calcular con precisión el movimiento de los cuerpos celestes.

## Características

- **Simulación física precisa**: Utiliza la ley de gravitación universal y el algoritmo de integración de Verlet
- **Visualización 3D interactiva**: Interfaz gráfica que permite rotar y explorar el sistema simulado
- **Soporte para múltiples cuerpos celestes**: Simula estrellas y planetas con propiedades físicas reales
- **Vector de traslación global**: Permite simular el movimiento del sistema completo a través del espacio
- **Trayectorias dinámicas**: Visualización de las órbitas con seguimiento automático

## Requisitos

- Python 3.6+
- NumPy
- Matplotlib

## Instalación

1. Clona este repositorio o descarga el archivo fuente
2. Instala las dependencias:
   ```
   pip install numpy matplotlib
   ```

## Uso

Ejecuta el script principal para iniciar la simulación del sistema solar simplificado:

```
python sistema_solar.py
```

### Personalización

Para modificar el sistema solar, edita la sección principal del código para añadir o modificar cuerpos celestes:

```
# Ejemplo para añadir un nuevo planeta
venus = Planeta(
    "Venus", 4.867e24, radio=6.052e6,
    x=1.082e11, y=0, z=0,
    vx=0, vy=35.02e3, vz=0
)

# Añade el nuevo planeta a las listas en la simulación y visualización
simular_sistema([sol, venus, tierra, marte], dt=360000, pasos=365*24, vector_traslacion=vector_traslacion)
visualizar_sistema_3d([sol, venus, tierra, marte], pasos_animacion=10000)
```

## Estructura del código

- `CuerpoCeleste` (ABC): Clase base abstracta para todos los objetos espaciales
- `Planeta`: Implementación para cuerpos que orbitan bajo influencia gravitacional
- `Estrella`: Implementación para cuerpos estelares (típicamente estáticos salvo por traslación global)
- `calcular_fuerza()`: Calcula la fuerza gravitacional entre dos cuerpos
- `simular_sistema()`: Ejecuta la simulación física para un conjunto de cuerpos
- `visualizar_sistema_3d()`: Muestra los resultados de la simulación con gráficos interactivos

## Algoritmo de integración de Verlet

El simulador utiliza el algoritmo de integración de Verlet para una mayor precisión en la simulación de órbitas:

```
# Actualización de posición
x += vx * dt + 0.5 * ax * dt**2

# Actualización de velocidad
vx += 0.5 * (ax_prev + ax) * dt
```

Este método conserva la energía en sistemas gravitacionales mejor que otros integradores como Euler.

## Ejemplos de simulación

El ejemplo predeterminado simula:
- Sol (estrella central)
- Tierra (órbita a 1 UA)
- Marte (órbita a 1.52 UA)

Todo el sistema también se mueve a través del espacio con una velocidad de 5 km/s en X, 8 km/s en Y y 15 km/s en Z.
