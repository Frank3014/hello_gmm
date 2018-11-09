/*
 * main.cpp
 *
 *  Created on: Nov 4, 2018
 *      Author: user
 */

#include <random>
#include "GMM.h"
#include "KMeans.h"

int main()
{
	int trainCount = 500;
	MatrixXd trainData;
	trainData.resize(500, 2);

	std::default_random_engine generator;

	std::normal_distribution<double> dis11(0.0, 1.0);
	std::normal_distribution<double> dis12(5.0, 3.0);

	std::normal_distribution<double> dis21(5, 2.0);
	std::normal_distribution<double> dis22(10, 4.0);

	std::normal_distribution<double> dis31(10.0, 3.0);
	std::normal_distribution<double> dis32(15.0, 6.0);

	for (int i = 0; i < trainCount; i += 10) {
		trainData(i, 0) = dis11(generator);
		trainData(i, 1) = dis12(generator);

		trainData(i + 1, 0) = dis11(generator);
		trainData(i + 1, 1) = dis12(generator);

		trainData(i + 2, 0) = dis21(generator);
		trainData(i + 2, 1) = dis22(generator);

		trainData(i + 3, 0) = dis21(generator);
		trainData(i + 3, 1) = dis22(generator);

		trainData(i + 4, 0) = dis21(generator);
		trainData(i + 4, 1) = dis22(generator);

		trainData(i + 5, 0) = dis21(generator);
		trainData(i + 5, 1) = dis22(generator);

		trainData(i + 6, 0) = dis31(generator);
		trainData(i + 6, 1) = dis32(generator);

		trainData(i + 7, 0) = dis31(generator);
		trainData(i + 7, 1) = dis32(generator);

		trainData(i + 8, 0) = dis31(generator);
		trainData(i + 8, 1) = dis32(generator);

		trainData(i + 9, 0) = dis31(generator);
		trainData(i + 9, 1) = dis32(generator);
	}

	KMeans kmeans;
	MatrixXd means;
	kmeans.Train(trainData, 3, means);
	kmeans.PrintModel();

	GMM gmm;
	gmm.Init(trainData.cols(), 3, means);
	gmm.Train(trainData);
	gmm.PrintModel();

	return 0;
}
