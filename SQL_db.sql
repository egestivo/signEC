CREATE DATABASE reconocimiento_senas;

USE reconocimiento_senas;

CREATE TABLE palabra (
    id INT AUTO_INCREMENT PRIMARY KEY,
    texto VARCHAR(100) NOT NULL,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE puntuacion (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre_usuario VARCHAR(50) NOT NULL,
    puntuacion INT NOT NULL,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP
);


SELECT * FROM palabra;
SELECT * FROM puntuacion;