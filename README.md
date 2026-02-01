# AA_MaestriaUEES

Repositorio para la materia de **Aprendizaje Automatico** - Maestria en Inteligencia Artificial, UEES.

---
Estudiante: Ingeniero Gonzalo Mejia Alcivar
Docente: Ingeniera GLADYS MARIA VILLEGAS RUGEL
Fecha de Ultima Actualizacion: 01 Febrero 2026

## Dataset: Social Network Ads

### Dominio

El dataset **Social_Network_Ads.csv** pertenece al dominio de **marketing digital y comportamiento del consumidor en redes sociales**. Contiene informacion de 400 usuarios de una red social, recopilada con el objetivo de predecir si un usuario comprara un producto despues de ver un anuncio publicitario en la plataforma.

Este tipo de datos es comun en estrategias de **publicidad dirigida (targeted advertising)**, donde las empresas buscan identificar el perfil de clientes con mayor probabilidad de compra para optimizar sus campanas y presupuesto publicitario.

### Descripcion de Variables

| Variable | Tipo | Descripcion |
|---|---|---|
| `User ID` | Identificador | ID unico del usuario |
| `Gender` | Categorica | Genero del usuario (Male / Female) |
| `Age` | Numerica | Edad del usuario (rango: 18 - 60) |
| `EstimatedSalary` | Numerica | Salario estimado del usuario en dolares (rango: 15,000 - 150,000) |
| `Purchased` | Categorica (Target) | Indica si el usuario compro el producto (1 = Si, 0 = No) |

### Objetivo

El problema de clasificacion binaria consiste en predecir la variable **Purchased** a partir de las caracteristicas demograficas del usuario (edad y salario estimado). Esto permite a los equipos de marketing:

- Segmentar audiencias de manera mas efectiva.
- Dirigir anuncios a usuarios con mayor probabilidad de conversion.
- Reducir costos de adquisicion de clientes.

