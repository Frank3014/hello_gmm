/*
 * KMeans.cpp
 *
 *  Created on: Nov 8, 2018
 *      Author: user
 */

#include "KMeans.h"

KMeans::KMeans() {
	// TODO Auto-generated constructor stub

}

KMeans::~KMeans() {
	// TODO Auto-generated destructor stub
}


int KMeans::Train(const MatrixXd data, int cateCount, MatrixXd &means){

	// Initialize
	Init(data, cateCount);

	while(true){
		// Classification
		this->m_nextCost = 0;
		this->m_meansCount.setZero(this->m_meansCount.rows());
		this->m_nextMeans.setZero(this->m_nextMeans.rows(), this->m_nextMeans.cols());
		for(int i = 0; i < data.rows(); ++i){
			int label;
			this->m_nextCost += Classification(data.row(i), label);
			this->m_meansCount(label) += 1;
			m_nextMeans.row(label) += data.row(i);
		}
		this->m_nextCost /= data.rows();

		// Update
		for(int i = 0; i < this->m_nextMeans.rows(); ++i){
			this->m_nextMeans.row(i) /= this->m_meansCount(i);
		}

		// Terminal flag
		this->m_trainIndex += 1;
		if((fabs(this->m_cost - this->m_nextCost) < this->m_endError * this->m_cost) || (this->m_trainIndex > this->m_maxCount)){
			break;
		}

		this->m_means = this->m_nextMeans;
		this->m_cost = this->m_nextCost;
	}

	means.resize(this->m_means.rows(), this->m_means.cols());
	means = this->m_means;

	return 0;
}

int KMeans::PrintModel(){

	printf("<KMeans>\n");
	printf("<DimCount> %d <DimCount>\n", m_dimCount);
	printf("<CateCount> %d <DimCount>\n", m_cateCount);

	int totalCount = 0;
	for(int i = 0; i < this->m_meansCount.rows(); ++i){
		totalCount += this->m_meansCount(i);
	}

	printf("<Prior>\n");
	for(int i = 0; i < this->m_meansCount.rows(); ++i){
		printf("%f ", this->m_meansCount(i)/totalCount);
	}
	printf("\n<Prior>\n");

	printf("<Mean>\n");
	for(int i = 0; i < this->m_means.rows(); ++i){
		for(int j = 0; j < this->m_means.cols(); ++j){
			printf("%f ", m_means(i, j));
		}
		printf("\n");
	}
	printf("<Mean>\n");

	return 0;
}

int KMeans::Init(const MatrixXd data, int cateCount){

	this->m_cateCount = cateCount;
	this->m_dimCount = data.cols();
	this->m_means.resize(cateCount, data.cols());
	this->m_nextMeans.resize(cateCount, data.cols());
	this->m_meansCount.resize(cateCount);

	std::default_random_engine generator;
	std::uniform_int_distribution<int> dis(0, data.rows());

	for(int i = 0; i < cateCount; ++i){
		this->m_means.row(i) = data.row(dis(generator));
	}

	this->m_nextMeans = this->m_means;

	this->m_cost = 0;
	this->m_nextCost = 0;

	this->m_endError = 0.001;
	this->m_maxCount = 100000;

	return 0;
}

double KMeans::Classification(const VectorXd sample, int &label){
	double minDist = CalcDistance(this->m_means.row(0), sample);
	label = 0;
	for (int i = 1; i < this->m_cateCount; ++i)
	{
		double tmpDist = CalcDistance(this->m_means.row(i), sample);
		if(tmpDist < minDist )
		{
			minDist = tmpDist;
			label = i;
		}
	}

	return minDist;
}


double KMeans::CalcDistance(const VectorXd means, const VectorXd sample){
	double tmp = 0;

	for(int i = 0; i < means.rows(); ++i){
		tmp += pow((sample(i) - means(i)), 2);
	}

	return sqrt(tmp);
}
