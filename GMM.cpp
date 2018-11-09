/*
 * GMM.cpp
 *
 *  Created on: Nov 1, 2018
 *      Author: user
 */

#include "GMM.h"
#include <math.h>
#include <stdio.h>

//#define DEBUG_PRINT

GMM::GMM() {
	// TODO Auto-generated constructor stub

}

GMM::~GMM() {
	// TODO Auto-generated destructor stub
}


int GMM::Init(int dim, int cate){
	this->m_dimCount = dim;
	this->m_cateCount = cate;

	this->m_priors.resize(this->m_cateCount);
	this->m_priors.setOnes(this->m_cateCount);
	this->m_priors = this->m_priors / this->m_cateCount;
	this->m_tmp_priors.resize(this->m_cateCount);
	this->m_tmp_priors = this->m_priors;

	this->m_means.resize(this->m_cateCount, this->m_dimCount);
	this->m_means.setOnes(this->m_cateCount, this->m_dimCount);
	for(int i = 0; i < this->m_cateCount; ++i){
		this->m_means.row(i) = this->m_means.row(i) * i * 25;
	}
	this->m_tmp_means.resize(this->m_cateCount, this->m_dimCount);
	this->m_tmp_means = this->m_means;

	this->m_vars.resize(this->m_cateCount, this->m_dimCount);
	this->m_vars.setOnes(this->m_cateCount, this->m_dimCount);
	this->m_vars = this->m_vars * 3;
	this->m_tmp_vars.resize(this->m_cateCount, this->m_dimCount);
	this->m_tmp_vars = this->m_vars;

	this->m_maxCount = 100000;

	this->m_minPriorErr = 0.000001;
	this->m_minMeanErr = 0.000001;
	this->m_minVarErr = 0.000001;

	return 0;
}

int GMM::Init(int dim, int cate, const MatrixXd means){
	this->m_dimCount = dim;
	this->m_cateCount = cate;

	this->m_priors.resize(this->m_cateCount);
	this->m_priors.setOnes(this->m_cateCount);
	this->m_priors = this->m_priors / this->m_cateCount;
	this->m_tmp_priors.resize(this->m_cateCount);
	this->m_tmp_priors = this->m_priors;

	this->m_means.resize(this->m_cateCount, this->m_dimCount);
	this->m_means = means;
	this->m_tmp_means.resize(this->m_cateCount, this->m_dimCount);
	this->m_tmp_means = this->m_means;

	this->m_vars.resize(this->m_cateCount, this->m_dimCount);
	this->m_vars.setOnes(this->m_cateCount, this->m_dimCount);
	this->m_vars = this->m_vars * 3;
	this->m_tmp_vars.resize(this->m_cateCount, this->m_dimCount);
	this->m_tmp_vars = this->m_vars;

	this->m_maxCount = 100000;

	this->m_minPriorErr = 0.000001;
	this->m_minMeanErr = 0.000001;
	this->m_minVarErr = 0.000001;

	return 0;
}

int GMM::Train(const MatrixXd data){

	// Probability of sample belonging to category
	this->m_gamma.resize(data.rows(), this->m_cateCount);

	this->m_trainIndex = 0;

	while(true){
		// E step
		this->m_gamma.setZero(data.rows(), this->m_cateCount);
		for(int i = 0; i < data.rows(); ++i){
			double sum = 0.0;
			for(int j = 0; j < this->m_cateCount; ++j){
				sum += this->m_priors(j) * Gauss(this->m_means.row(j), this->m_vars.row(j), data.row(i));
			}

			for(int j = 0; j < this->m_cateCount; ++j){
				this->m_gamma(i, j) = this->m_priors(j) * Gauss(this->m_means.row(j), this->m_vars.row(j), data.row(i)) / sum;
			}

#ifdef DEBUG_PRINT
			printf("gamma[%d] : (", i);
			for(int j = 0; j < this->m_cateCount; ++j){
				printf(" %f", data(i, j));
			}
			printf(" ) : (");
			for(int j = 0; j < this->m_cateCount; ++j){
				printf(" %f", this->m_gamma(i, j));
			}
			printf(" )\n");
#endif
		}

		// M step
		for(int i = 0; i < this->m_cateCount; ++i){
			double sum = 0.0;
			for(int j = 0; j < data.rows(); ++j){
				sum += this->m_gamma(j, i);
			}

			VectorXd means(data.cols());
			VectorXd vars(data.cols());
			means.setZero(data.cols());
			vars.setZero(data.cols());
			for(int j = 0; j < data.rows(); ++j){
				means += this->m_gamma(j, i) * data.row(j);
				for(int k = 0; k < data.cols(); ++k){
					vars(k) += (this->m_gamma(j, i) * pow((data(j, k) - this->m_means(i, k)), 2));
				}
			}

			this->m_tmp_priors(i) = sum / data.rows();
			this->m_tmp_means.row(i) = means / sum;
			for(int j = 0; j < this->m_tmp_vars.cols(); ++j){
				this->m_tmp_vars(i, j) = sqrt(vars(j) / sum);
			}
		}

		// Terminal flag
		bool priors_vanished_flag = true;
		for(int i = 0; i < this->m_priors.rows(); ++i){
			if(fabs(this->m_priors(i) - this->m_tmp_priors(i)) > m_minPriorErr){
				priors_vanished_flag = false;
			}
		}

		bool meas_vanished_flag = true;
		for(int i = 0; i < this->m_means.rows(); ++i){
			for(int j = 0; j < this->m_means.cols(); ++j){
				if(fabs(this->m_means(i, j) - this->m_tmp_means(i, j)) > m_minMeanErr){
					meas_vanished_flag = false;
				}
			}
		}

		bool var_vanished_flag = true;
		for(int i = 0; i < this->m_vars.rows(); ++i){
			for(int j = 0; j < this->m_vars.cols(); ++j){
				if(fabs(this->m_vars(i, j) - this->m_tmp_vars(i, j)) > m_minVarErr){
					var_vanished_flag = false;
				}
			}
		}

		// update
		this->m_priors = this->m_tmp_priors;
		this->m_means = this->m_tmp_means;
		this->m_vars = this->m_tmp_vars;


		// Terminal
		if(((true == priors_vanished_flag) && (true == meas_vanished_flag) && (true == var_vanished_flag)) || (this->m_trainIndex > m_maxCount)){
			break;
		}

		this->m_trainIndex += 1;
	}

	return 0;
}

int GMM::PrintModel(){

	printf("<GMM>\n");
	printf("<DimCount> %d <DimCount>\n", m_dimCount);
	printf("<CateCount> %d <DimCount>\n", m_cateCount);

	printf("<Prior>\n");
	for(int i = 0; i < this->m_priors.rows(); ++i){
		printf("%f ", m_priors(i));
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

	printf("<Variance>\n");
	for(int i = 0; i < this->m_vars.rows(); ++i){
		for(int j = 0; j < this->m_vars.cols(); ++j){
			printf("%f ", m_vars(i, j));
		}
		printf("\n");
	}
	printf("<Variance>\n");

	return 0;
}

double GMM::Gauss(const VectorXd means, const VectorXd vars, const VectorXd sample){
	double p = 1.0;

	for(int i = 0; i < means.rows(); ++i){
		p *= ((1.0 / (sqrt(2 * M_PI) * vars(i))) * exp(-0.5 * pow(sample(i) - means(i), 2) / pow(vars(i), 2)));
	}

	return p;
}
