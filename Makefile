# Makefile para generar el ejecutable de una red neuronal MLP para clasificación

CPP = g++
CPPFLAGS = -Wall
OBJECT = -c
NAME = -o

destino: ejecutable clean

ejecutable: main perceptronMulticapa
	@$(CPP) $(CPPFLAGS) main.o perceptronMulticapa.o $(NAME) mlpClassification.x
	@echo Creando mlpClassification.x

main: main.cpp
	@$(CPP) $(CPPFLAGS) $(OBJECT) main.cpp
	@echo Creando main.o

perceptronMulticapa: perceptronMulticapa.hpp perceptronMulticapa.cpp
	@$(CPP) $(CPPFLAGS) $(OBJECT) perceptronMulticapa.cpp
	@echo Creando perceptronMulticapa.o

clean:
	@rm *.o
	@echo Borrando archivos *.o
