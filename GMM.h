/*
 * GMM.h
 *
 *  Created on: Nov 1, 2018
 *      Author: user
 */

#ifndef GMM_H_
#define GMM_H_

#include <Eigen/Core>
using namespace Eigen;


/*----------------------------------------------------------------------------------------------------
 * node :
 * 		1. It is need to determine the number of classifications in GMM.
 * 		2. GMM is sensitive to initialization the center of mass for each classification.
 * 		3. The optimal solution may not be obtained if the mean initialization is unreasonable.
 * 		4. GMM usually needs K-means or other algorithm to calculate the mean initialize value.
 *---------------------------------------------------------------------------------------------------*/
class GMM {
public:
	GMM();
	virtual ~GMM();

public:
	int Init(int dim, int cate);
	int Init(int dim, int cate, const MatrixXd means);

	int Train(const MatrixXd data);

	int PrintModel();

private:
	// Calculate Probability of Gauss distribution
	double Gauss(const VectorXd means, const VectorXd vars, const VectorXd sample);

private:
	int m_dimCount;				// sample dimension
	int m_cateCount;			// category number

	VectorXd m_priors;			// mixing coefficient
	MatrixXd m_means;			// mean vector
	MatrixXd m_vars;			// variance vector

	VectorXd m_tmp_priors;		// mixing coefficient
	MatrixXd m_tmp_means;		// mean vector
	MatrixXd m_tmp_vars;		// variance vector

	MatrixXd m_gamma;			// mean vector

	long int m_trainIndex;		// train index
	long int m_maxCount;		// max count for train

	double m_minPriorErr;		// The stopping criterion regarding the error
	double m_minMeanErr;		// The stopping criterion regarding the error
	double m_minVarErr;			// The stopping criterion regarding the error
};

#endif /* GMM_H_ */
