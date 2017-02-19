/*********************************************************************
 * File  : PerceptronMulticapa.h
 * Date  : 2016
 *********************************************************************/


#ifndef _PERCEPTRONMULTICAPA_H_
#define _PERCEPTRONMULTICAPA_H_

namespace imc {

// Estructuras para la red neuronal
// ---------------------
struct Neurona {
	double x;  /* Salida producida por la neurona (out_j^h)*/
	double dX; /* Derivada de la salida producida por la neurona (delta_j)*/
	std::vector<double> w;            /* Vector de pesos de entrada (w_{ji}^h)*/
	std::vector<double> deltaW;       /* Cambio a aplicar a cada peso de entrada (\Delta_{ji}^h (t))*/
	std::vector<double> ultimoDeltaW; /* Último cambio aplicada a cada peso (\Delta_{ji}^h (t-1))*/
	std::vector<double> wCopia;       /* Copia de los pesos de entrada */
};

struct Capa {
	int nNumNeuronas; /* Número de neuronas de la capa*/
	int tipo;         /* Tipo de la capa (0=> sigmoide, 1=> softmax)*/
	std::vector<Neurona> pNeuronas; /* Vector con las neuronas de la capa*/
};

struct Datos {
	int nNumEntradas; /* Número de entradas */
	int nNumSalidas;  /* Número de salidas */
	int nNumPatrones; /* Número de patrones */
	std::vector<std::vector<double> > entradas; /* Matriz con las entradas del problema */
	std::vector<std::vector<double> > salidas;  /* Matriz con las salidas del problema */
};

class PerceptronMulticapa {
private:
	int nNumCapas; /* Número de capas total en la red */
	std::vector<Capa> pCapas; /* Vector con cada una de las capas */

	// Valores de parámetros de la red neuronal
	double dEta;        // Tasa de aprendizaje
	double dMu;         // Factor de momento
	bool   bSesgo;      // ¿Van a tener sesgo las neuronas?
	bool   bOnline;     // ¿El aprendizaje va a ser online? (true->online,false->offline)

	// Liberar memoria para las estructuras de datos
	void liberarMemoria();

	// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
	void pesosAleatorios();

	// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
	void alimentarEntradas(const std::vector<double> &entrada);

	// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
	void recogerSalidas(std::vector<double> &salida);

	// Hacer una copia de todos los pesos (copiar w en copiaW)
	void copiarPesos();

	// Restaurar una copia de todos los pesos (copiar copiaW en w)
	void restaurarPesos();

	// Calcular y propagar las salidas de las neuronas, desde la segunda capa hasta la última
	void propagarEntradas();

	// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
	// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
	double calcularErrorSalida(const std::vector<double> &objetivo, const int &funcionError);

	// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
	// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
	void retropropagarError(const std::vector<double> &objetivo, const int &funcionError);

	// Acumular los cambios producidos por un patrón en deltaW
	void acumularCambio();

	// Actualizar los pesos de la red, desde la segunda capa hasta la última
	void ajustarPesos();

	// Imprimir la red, es decir, todas las matrices de pesos
	void imprimirRed();

	// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
	// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
	// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
	// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
	// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
	void simularRed(const std::vector<double> &entrada, const std::vector<double> &objetivo, const int &funcionError);

public:

	// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
	PerceptronMulticapa();

	// DESTRUCTOR: liberar memoria
	~PerceptronMulticapa();

	// Métodos observadores de los parámetros de la red neuronal

	inline bool isSesgo() const {
		return this->bSesgo;
	}

	inline double getEta() const {
		return this->dEta;
	}

	inline double getMu() const {
		return this->dMu;
	}

	inline bool isOnline() const {
		return this->bOnline;
	}

	// Métodos modificadores de los parámetros de la red neuronal

	inline void setSesgo(const bool &sesgo) {
		this->bSesgo = sesgo;
	}

	inline void setEta(const double &eta) {
		this->dEta = eta;
	}

	inline void setMu(const double &mu) {
		this->dMu = mu;
	}

	inline void setOnline(const bool &online) {
		this->bOnline = online;
	}

	// Reservar memoria para las estructuras de datos
    // nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
    // Rellenar vector Capa* pCapas
	int inicializar(const int &nl, const std::vector<int> &npl, const bool &bSigmoideCapaSalida);

	// Leer una matriz de datos a partir de un nombre de fichero y devolverla
	Datos* leerDatos(const char * archivo);

	// Probar la red con un conjunto de datos y devolver el error MSE cometido
	// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
	double test(Datos* pDatosTest, const int &funcionError);

	// Probar la red con un conjunto de datos y devolver el error CCR cometido
	double testClassification(Datos* pDatosTest);

	// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
	// Si es offline, después de pasar por ellos hay que ajustar pesos. Sino, ya se ha ajustado en cada patrón
	void entrenar(Datos* pDatosTrain, const int &funcionError);

	// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
    // Una vez terminado, probar como funciona la red en pDatosTest
    // Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
	// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
	void ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, const int &maxiter, double &errorTrain, double &errorTest, double &ccrTrain, double &ccrTest, const int &funcionError);

};

};

#endif
