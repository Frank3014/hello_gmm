/*
 * KMeans.h
 *
 *  Created on: Nov 8, 2018
 *      Author: user
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include <Eigen/Core>
using namespace Eigen;

class KMeans {
public:
	KMeans();
	virtual ~KMeans();

	int Train(const MatrixXd data, int cateCount, MatrixXd &means);
	int PrintModel();

private:
	int Init(const MatrixXd data, int cateCount);
	double CalcDistance(const VectorXd means, const VectorXd sample);
	double Classification(const VectorXd sample, int &label);

private:
	int m_dimCount;				// sample dimension
	int m_cateCount;			// category number

	MatrixXd m_means;			// mean vector
	MatrixXd m_nextMeans;		// mean vector
	VectorXd m_meansCount;		// mean vector

	double m_cost;				// current cost
	double m_nextCost;			// next cost

	long int m_trainIndex;		// train index
	long int m_maxCount;		// max count for train

	double m_endError;			// The stopping criterion regarding the error
};

#endif /* KMEANS_H_ */
