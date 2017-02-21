/*********************************************************************
 * File  : perceptronMulticapa.cpp
 * Date  : 2016
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>
#include <vector>
#include <time.h>

// Inclusión del archivo de cabecera de PerceptrónMulticapa
#include "perceptronMulticapa.hpp"

// ------------------------------
// Obtener un número entero aleatorio en el intervalo [Low,High]
int enteroAleatorio(const int &Low, const int &High)
{
	return Low + (rand() % (int)(High - Low + 1));
}

// ------------------------------
// Obtener un número real aleatorio en el intervalo [Low,High]
double realAleatorio(const double &Low, const double &High)
{
	double valor = (double) rand() / RAND_MAX;
	return Low + valor * (High-Low);
}

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
imc::PerceptronMulticapa::PerceptronMulticapa()
{
	this->dEta = 0.1;
	this->dMu = 0.9;
	this->bSesgo = false;
	this->nNumCapas = 3;
	this->bOnline = false;
}

// Reservar memoria para las estructuras de datos
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// Rellenar vector Capa* pCapas
int imc::PerceptronMulticapa::inicializar(const int &nl, const std::vector<int> &npl, const bool &bSigmoideCapaSalida) {

	// Se reserva espacio para el nº de capas de la red neuronal
	this->nNumCapas = nl;
	this->pCapas.resize(nl);

	// Se reserva espacio para el nº de neuronas por capa
	for(int h=0; h<nl; h++) {
		this->pCapas[h].nNumNeuronas = npl[h];
		this->pCapas[h].pNeuronas.resize(npl[h]);

		// Por defecto en principio todas las neuronas de capas actúan como sigmoide
		this->pCapas[h].tipo = 0;

		// Se reserva espacio para las salidas y pesos en capa oculta y de salida
		// (No es necesario en la capa de entrada)
		if (h > 0) {
			for(int j=0; j<npl[h]; j++) {
				this->pCapas[h].pNeuronas[j].w.resize(npl[h-1] + this->bSesgo);
				this->pCapas[h].pNeuronas[j].deltaW.resize(npl[h-1] + this->bSesgo);
				this->pCapas[h].pNeuronas[j].ultimoDeltaW.resize(npl[h-1] + this->bSesgo);
				this->pCapas[h].pNeuronas[j].wCopia.resize(npl[h-1] + this->bSesgo);
			}
		}
	}

	// Si se ha establecido, las neuronas de la última capa serán de tipo Softmax
	if (bSigmoideCapaSalida)
		this->pCapas[this->nNumCapas-1].tipo = 1;

	return EXIT_SUCCESS;
}


// ------------------------------
// DESTRUCTOR: liberar memoria
imc::PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void imc::PerceptronMulticapa::liberarMemoria() {

	for(int h=0; h<this->nNumCapas; h++) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			this->pCapas[h].pNeuronas[j].w.clear();
			this->pCapas[h].pNeuronas[j].deltaW.clear();
			this->pCapas[h].pNeuronas[j].ultimoDeltaW.clear();
			this->pCapas[h].pNeuronas[j].wCopia.clear();
		}
		this->pCapas[h].pNeuronas.clear();
	}
	this->pCapas.clear();
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void imc::PerceptronMulticapa::pesosAleatorios() {

	for(int h=1; h<this->nNumCapas; h++)
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++)
			for(int i=0; i<this->pCapas[h-1].nNumNeuronas + this->bSesgo; i++)
				this->pCapas[h].pNeuronas[j].w[i] = realAleatorio(-1,1);
}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void imc::PerceptronMulticapa::alimentarEntradas(const std::vector<double> &input) {

	for(int j=0; j<pCapas[0].nNumNeuronas; j++)
		this->pCapas[0].pNeuronas[j].x = input[j];
}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void imc::PerceptronMulticapa::recogerSalidas(std::vector<double> &output) {

	for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
		output[j] = this->pCapas[this->nNumCapas-1].pNeuronas[j].x;
}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void imc::PerceptronMulticapa::copiarPesos() {

	for(int h=0; h<this->nNumCapas; h++)
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++)
			this->pCapas[h].pNeuronas[j].wCopia = this->pCapas[h].pNeuronas[j].w;
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void imc::PerceptronMulticapa::restaurarPesos() {

	for(int h=0; h<this->nNumCapas; h++)
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++)
			this->pCapas[h].pNeuronas[j].w = this->pCapas[h].pNeuronas[j].wCopia;
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la segunda capa hasta la última
void imc::PerceptronMulticapa::propagarEntradas() {

	// Valor de salida de la neurona j al propagarse
	double salida = 0.0;

	// Sumatorio de las salidas softmax de todas las neuronas de la última capa
	double sumatorioSoftmax = 0.0;

	for(int h=1; h<this->nNumCapas; h++) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[h-1].nNumNeuronas; i++)
				salida += this->pCapas[h].pNeuronas[j].w[i] * this->pCapas[h-1].pNeuronas[i].x;

			// Se incluye el sesgo en la función sigmoide o softmax si está activo
			if (this->bSesgo)
				salida += this->pCapas[h].pNeuronas[j].w[this->pCapas[h-1].nNumNeuronas];

			// Se realiza el primer cálculo de la función softmax
			// Se añade al sumatorio el valor de todas las salidas softmax
			// Si se activa la función softmax, la variable salida es positiva
			if (h == this->nNumCapas-1 and this->pCapas[this->nNumCapas-1].tipo == 1) {
				this->pCapas[h].pNeuronas[j].x = exp(salida);
				sumatorioSoftmax += this->pCapas[h].pNeuronas[j].x;
			// Se realiza la función sigmoide
			// La variable salida es negativa con la sigmoide
			}else
				this->pCapas[h].pNeuronas[j].x = 1 / (1 + exp(-salida));

			// Se reinicializa el valor de la variable salida
			salida = 0.0;
		}
	}

	// Si está activada la función softmax, nos queda un último cálculo por hacer
	if (this->pCapas[this->nNumCapas-1].tipo == 1) {
		for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
			this->pCapas[this->nNumCapas-1].pNeuronas[j].x /= sumatorioSoftmax;
	}
}

// ------------------------------
// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double imc::PerceptronMulticapa::calcularErrorSalida(const std::vector<double> &target, const int &funcionError) {

	// Variable con el error cometido (Entropía cruzada o MSE)
	double error = 0.0;

	// Función de error Entropía Cruzada
	if (funcionError) {
		for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
			error -= target[j] * log(this->pCapas[this->nNumCapas-1].pNeuronas[j].x);
	// Función de error MSE
	}else{
		for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
			error += pow(target[j] - this->pCapas[this->nNumCapas-1].pNeuronas[j].x,2);
	}

	// Se ha de dividir dicho error calculado entre el número de neuronas de salida
	return error / this->pCapas[this->nNumCapas-1].nNumNeuronas;
}

// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::PerceptronMulticapa::retropropagarError(const std::vector<double> &objetivo, const int &funcionError) {

	// Si la última capa contiene neuronas con función Sigmoide...
	if (this->pCapas[this->nNumCapas-1].tipo == 0) {
		if (funcionError)
			// Función de error Entropía Cruzada
			for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
				this->pCapas[this->nNumCapas-1].pNeuronas[j].dX = -(objetivo[j] / this->pCapas[this->nNumCapas-1].pNeuronas[j].x) * this->pCapas[this->nNumCapas-1].pNeuronas[j].x * (1 - this->pCapas[this->nNumCapas-1].pNeuronas[j].x);
		else
			// Función de error MSE
			for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++)
				this->pCapas[this->nNumCapas-1].pNeuronas[j].dX = -(objetivo[j] - this->pCapas[this->nNumCapas-1].pNeuronas[j].x) * this->pCapas[this->nNumCapas-1].pNeuronas[j].x * (1 - this->pCapas[this->nNumCapas-1].pNeuronas[j].x);
	// Si la última capa contiene neuronas con función Softmax...
	}else{
		// Sumatorio de las salidas softmax de todas las neuronas de la última capa
		double sumatorioSoftmax = 0.0;

		// Contienen las dos partes por separado de los cálculos necesarios
		double primerCalculo, segundoCalculo;

		for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[this->nNumCapas-1].nNumNeuronas; i++) {
				// Función de error Entropía Cruzada
				if (funcionError)
					primerCalculo = objetivo[i] / this->pCapas[this->nNumCapas-1].pNeuronas[i].x;
				// Función de error MSE
				else
					primerCalculo = objetivo[i] - this->pCapas[this->nNumCapas-1].pNeuronas[i].x;

				if (i == j)
					segundoCalculo = this->pCapas[this->nNumCapas-1].pNeuronas[j].x * (1 - this->pCapas[this->nNumCapas-1].pNeuronas[i].x);
				else
					segundoCalculo = this->pCapas[this->nNumCapas-1].pNeuronas[j].x * (-this->pCapas[this->nNumCapas-1].pNeuronas[i].x);

				sumatorioSoftmax -= primerCalculo * segundoCalculo;
			}
			this->pCapas[this->nNumCapas-1].pNeuronas[j].dX = sumatorioSoftmax;
			sumatorioSoftmax = 0.0;
		}
	}

	// Contiene el sumatorio calculado para cada una de las neuronas
	double sumatorio = 0.0;

	// Se retropaga el error por las diferentes capas
	for(int h=this->nNumCapas-2; h>0; h--) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[h+1].nNumNeuronas; i++)
				sumatorio += this->pCapas[h+1].pNeuronas[i].w[j] * this->pCapas[h+1].pNeuronas[i].dX;

			this->pCapas[h].pNeuronas[j].dX = sumatorio * this->pCapas[h].pNeuronas[j].x * (1 - this->pCapas[h].pNeuronas[j].x);
			sumatorio = 0.0;
		}
	}
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void imc::PerceptronMulticapa::acumularCambio() {

	for(int h=1; h<this->nNumCapas; h++) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[h-1].nNumNeuronas; i++)
				this->pCapas[h].pNeuronas[j].deltaW[i] += this->pCapas[h].pNeuronas[j].dX * this->pCapas[h-1].pNeuronas[i].x;

			if (this->bSesgo)
				// La última posición del vector deltaW contiene el sesgo, si es que existe
				this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas] += this->pCapas[h].pNeuronas[j].dX;
		}
	}
}

// ------------------------------
// Actualizar los pesos de la red, desde la segunda capa hasta la última
void imc::PerceptronMulticapa::ajustarPesos() {

	for(int h=1; h<this->nNumCapas; h++) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[h-1].nNumNeuronas; i++) {
				this->pCapas[h].pNeuronas[j].w[i] += -(this->dEta * this->pCapas[h].pNeuronas[j].deltaW[i]) - (this->dMu * (this->dEta * this->pCapas[h].pNeuronas[j].ultimoDeltaW[i]));
				this->pCapas[h].pNeuronas[j].ultimoDeltaW[i] = this->pCapas[h].pNeuronas[j].deltaW[i];
			}

			if (this->bSesgo) {
				this->pCapas[h].pNeuronas[j].w[this->pCapas[h-1].nNumNeuronas] += -(this->dEta * this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas]) - (this->dMu * (this->dEta * this->pCapas[h].pNeuronas[j].ultimoDeltaW[this->pCapas[h-1].nNumNeuronas]));
				this->pCapas[h].pNeuronas[j].ultimoDeltaW[this->pCapas[h-1].nNumNeuronas] = this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas];
			}
		}
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void imc::PerceptronMulticapa::imprimirRed() {

	// La capa de entrada no tiene pesos asociados
	for(int h=1; h<this->nNumCapas; h++) {
		std::cout << "\n **********" << std::endl;
		std::cout << "  Capa <" << h << ">" << std::endl;
		std::cout << " **********" << std::endl;

		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			std::cout << "\n # Neurona <" << j << ">" << std::endl;
			std::cout << "\n  > Pesos: ";

			for(int i=0; i<this->pCapas[h-1].nNumNeuronas + this->bSesgo; i++)
				std::cout << this->pCapas[h].pNeuronas[j].w[i] << " ";

			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

// ------------------------------
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::PerceptronMulticapa::simularRed(const std::vector<double> &entrada, const std::vector<double> &objetivo, const int &funcionError) {

	// Se realizan los diferentes pasos para la simulación de la red neuronal
	alimentarEntradas(entrada);
	propagarEntradas();
	retropropagarError(objetivo,funcionError);
	acumularCambio();

	// Sólo se ajustan los pesos para cada patrón en el algoritmo On-line
	// Sólo se restablecen los valores de delta para cada patrón en el algoritmo On-line
	if (this->bOnline) {
		ajustarPesos();

		// Se establecen los valores de delta a 0
		for(int h=1; h<this->nNumCapas; h++) {
			for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
				for(int i=0; i<this->pCapas[h-1].nNumNeuronas; i++)
					this->pCapas[h].pNeuronas[j].deltaW[i] = 0.0;

				if (this->bSesgo)
					this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas] = 0.0;
			}
		}
	}
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
imc::Datos* imc::PerceptronMulticapa::leerDatos(const char * archivo) {

	// Estructura con los datos leídos que se devuelve
	imc::Datos * pDatos = new imc::Datos;

	// Se abre el fichero de texto
	std::ifstream f(archivo);

	// Se lee el nº de entradas, salidas y patrones de la red neuronal
	f >> pDatos->nNumEntradas >> pDatos->nNumSalidas >> pDatos->nNumPatrones;

	// Se reserva memoria para las matrices de entrada y salida
	pDatos->entradas.resize(pDatos->nNumPatrones);
	pDatos->salidas.resize(pDatos->nNumPatrones);

	for(int i=0; i<pDatos->nNumPatrones; i++) {
		pDatos->entradas[i].resize(pDatos->nNumEntradas);
		pDatos->salidas[i].resize(pDatos->nNumSalidas);
	}

	// Se procede a leer los valores de entrada y salida de patrones
	for(int i=0; i<pDatos->nNumPatrones; i++) {
		// Se incluyen las entradas en la matriz
		for(int j=0; j<pDatos->nNumEntradas; j++)
			f >> pDatos->entradas[i][j];

		// Se incluyen las salidas en la matriz
		for(int j=0; j<pDatos->nNumSalidas; j++)
			f >> pDatos->salidas[i][j];
	}

	// Se cierra el fichero de texto
	f.close();

	return pDatos;
}

// ------------------------------
// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
// Si es offline, después de pasar por ellos hay que ajustar pesos. Sino, ya se ha ajustado en cada patrón
void imc::PerceptronMulticapa::entrenar(Datos* pDatosTrain, const int &funcionError) {

	// Se establecen los valores de delta a 0
	for(int h=1; h<this->nNumCapas; h++) {
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++) {
			for(int i=0; i<this->pCapas[h-1].nNumNeuronas; i++)
				this->pCapas[h].pNeuronas[j].deltaW[i] = 0.0;

			if (this->bSesgo)
				this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas] = 0.0;
		}
	}

	for(int i=0; i<pDatosTrain->nNumPatrones; i++)
		simularRed(pDatosTrain->entradas[i], pDatosTrain->salidas[i], funcionError);

	// Una vez terminadas todas las iteraciones, hay que ajustar los pesos en la versión Off-line
	if (!this->bOnline)
		ajustarPesos();
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error MSE cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double imc::PerceptronMulticapa::test(Datos* pDatosTest, const int &funcionError) {

	double dAvgTestError = 0;
	for(int i=0; i<pDatosTest->nNumPatrones; i++) {
		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		dAvgTestError += calcularErrorSalida(pDatosTest->salidas[i],funcionError);
	}
	dAvgTestError /= pDatosTest->nNumPatrones;
	return dAvgTestError;
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error CCR cometido
double imc::PerceptronMulticapa::testClassification(Datos* pDatosTest) {

	// Variable con el valor del ccr
	double CCR = 0.0;

	// Matriz de confusión
	std::vector<std::vector<int> > matrizConfusion(pDatosTest->nNumSalidas,std::vector<int>(pDatosTest->nNumSalidas,0));

	for(int i=0; i<pDatosTest->nNumPatrones; i++) {

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();

		// Índice con la clase que se espera que se encuentre un patrón
        int indiceDeseado = 0;

        // Índice con la clase que se predice que se encuentre un patrón
        int indiceObtenido = 0;

        // Valor de la salida obtenida con mayor probabilidad de pertenencia
        double valorMaxObtenido = 0.0;

        // Se comprueba cómo de bien se han clasificado los patrones
        for(int j=0; j<this->pCapas[this->nNumCapas-1].nNumNeuronas; j++) {

        	// Se busca el índice de la clase que se espera que esté dicho patrón
            if(pDatosTest->salidas[i][j] == 1)
                indiceDeseado = j;

            // Se hace caso a la probabilidad de pertenencia mayor para calcular el índice de la clase
            // en la que se ha clasificado al patrón por predicción
            if(this->pCapas[this->nNumCapas-1].pNeuronas[j].x > valorMaxObtenido) {
            	valorMaxObtenido = this->pCapas[this->nNumCapas-1].pNeuronas[j].x;
                indiceObtenido = j;
            }
        }

        // Se añade el patrón a la matriz de confusión
        matrizConfusion[indiceDeseado][indiceObtenido]++;

        // Se incrementa el ccr si el indiceDeseado y Obtenido son iguales
        // Es decir, si el patrón predicho se ha clasificado correctamente
        if(indiceDeseado == indiceObtenido)
            CCR++;
        //else
        	//std::cout << "\n # Patrón mal clasificado: <" << i+1 << ">\n Pertenece a " << indiceDeseado << " - Clasificado como " << indiceObtenido << std::endl;
	}

	// Se imprime la matriz de confusión generada
	for(int i=0; i<pDatosTest->nNumSalidas; i++) {
		std::cout << "|";
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			std::cout << " " << matrizConfusion[i][j];
		std::cout << " |" << std::endl;
	}

	// Se calcula el CCR final y se devuelve
	return 100 * (CCR / pDatosTest->nNumPatrones);
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void imc::PerceptronMulticapa::ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, const int &maxiter, double &errorTrain, double &errorTest, double &ccrTrain, double &ccrTest, const int &funcionError)
{
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0.0;
	int numSinMejorar;

	// Comienza a contar el tiempo
	clock_t t;
	t = clock();

	// Aprendizaje del algoritmo
	do {

		entrenar(pDatosTrain,funcionError);

		double trainError = test(pDatosTrain,funcionError);
		// El 0.00001 es un valor de tolerancia, podría parametrizarse
		if(countTrain==0 or fabs(trainError - minTrainError) > 0.00001){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}else
			numSinMejorar++;

		if(numSinMejorar==50)
			countTrain = maxiter;

		countTrain++;

		std::cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << std::endl;
		//std::cout << "Iteración " << countTrain << "\t CCR de test: " << testClassification(pDatosTest) << std::endl;
		//std::cout << "Iteración " << countTrain << "\t | " << trainError << " | " << test(pDatosTest,funcionError) << " | " << testClassification(pDatosTrain) << " | " << testClassification(pDatosTest) << " |" << std::endl;

	} while ( countTrain<maxiter );

	// Termina de contar el tiempo
	t = clock() - t;
	std::cout << "\n # Tiempo en entrenar: " << ((float)t)/CLOCKS_PER_SEC << " segundos" << std::endl;;

	std::cout << "\nPesos de la red" << std::endl;
	std::cout << "===============" << std::endl;
	imprimirRed();

	std::cout << "Salida Esperada Vs Salida Obtenida (test)" << std::endl;
	std::cout << "=========================================" << std::endl;
	for(int i=0; i<pDatosTest->nNumPatrones; i++) {
		std::vector<double> prediccion(pDatosTest->nNumSalidas);

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			std::cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j]<< " \\\\ " ;
			//std::cout << prediccion[j]<< ";" ;
		std::cout << std::endl;
		prediccion.clear();

	}

	errorTest = test(pDatosTest,funcionError);
	errorTrain = minTrainError;

	std::cout << "\n # Entrenamiento - Matriz de confusión:" << std::endl;
	ccrTrain = testClassification(pDatosTrain);

	std::cout << "\n # Test - Matriz de confusión:" << std::endl;
	ccrTest = testClassification(pDatosTest);
}
