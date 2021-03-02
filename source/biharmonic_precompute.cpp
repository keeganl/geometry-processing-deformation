#include "biharmonic_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/normal_derivative.h>
#include <igl/invert_diag.h>
#include <Eigen/Sparse>
#include <igl/invert_diag.h>
#include <igl/on_boundary.h>
#include <igl/crouzeix_raviart_cotmatrix.h>
#include <igl/crouzeix_raviart_massmatrix.h>
#include <igl/sum.h>
#include <igl/adjacency_matrix.h>
#include <vector>
#include <fstream>

void biharmonic_precompute(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &b,
    igl::min_quad_with_fixed_data<double> &data)
{

#ifdef false
  // get mass matrix
  Eigen::SparseMatrix<double> massMatrix;
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, massMatrix);

  // get cotangent laplacian
  Eigen::SparseMatrix<double> cotLaplacian, Z;
  igl::cotmatrix(V, F, cotLaplacian);

   Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> C;
    {
      Eigen::Array<bool,Eigen::Dynamic,1> I;
      igl::on_boundary(F,I,C);
    }

  // Q or A is a (usually sparse) n x n positive semi-definite matrix of quadratic coefficients (Hessian)
  // to find massMatrix inverse need to use a solver for a sparse matrix
  Eigen::SparseMatrix<double> K, Q, M_inv, N;

  // this is the wrong implementation
  igl::invert_diag(massMatrix, M_inv);

  igl::normal_derivative(V, F, N);

  {
      std::vector<Eigen::Triplet<double>> ZIJV;
      for(int t =0;t<F.rows();t++)
      {
        for(int f =0;f<F.cols();f++)
        {
          if(C(t,f))
          {
            const int i = t+f*F.rows();
            for(int c = 1;c<F.cols();c++)
            {
              ZIJV.emplace_back(F(t,(f+c)%F.cols()),i,1);
            }
          }
        }
      }
      Z.resize(V.rows(),N.rows());
      Z.setFromTriplets(ZIJV.begin(),ZIJV.end());
      N = (Z*N).eval();
    }

  K = N + cotLaplacian;

  Q = K.transpose() * M_inv * K;

  igl::min_quad_with_fixed_precompute(Q, b, Eigen::SparseMatrix<double>(), true, data);


#else 
  std::cout << "This is the version from the paper" << std::endl;

  std::ofstream CFile("logs_c.txt");
  std::ofstream cotLFile("logs_cotLaplacian.txt");
  std::ofstream massMatrixFile("logs_massMatrix.txt");
  std::ofstream AdFile("logs_ad.txt");
  std::ofstream KFile("logs_k.txt");

  Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> C;
  {
    Eigen::Array<bool,Eigen::Dynamic,1> I;
    igl::on_boundary(F,I,C);
  }

  Eigen::SparseMatrix<double> Q, M, L, K, N, Z, M_inv, cotLaplacian, massMatrix;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> E;
	Eigen::Matrix<int, Eigen::Dynamic, 1> EMAP;
  igl::cotmatrix(V, F, cotLaplacian);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, massMatrix);

  // the crouzeix_ravariant versions seem to be num_of_unique_edges*num_of_unique_edges
  // i think they are higher precision as well
  igl::crouzeix_raviart_massmatrix(V, F, M, E, EMAP);


  /*
  Reference pg 7-11 in this 
  https://cims.nyu.edu/gcl/papers/wardetzky2007dqb.pdf

  Deconstruct this loop for the c_r_cotmatrix
   factor = 4.0;
      LI<<0,1,2,1,2,0,0,1,2,1,2,0;
      LJ<<1,2,0,0,1,2,0,1,2,1,2,0;
      LV<<2,0,1,2,0,1,2,0,1,2,0,1;
  for(int f=0;f<m;f++)
  {
    for(int c = 0;c<k;c++)
    {
      LIJV.emplace_back(
        EMAP(F2E(f,LI(c)), 0),
        EMAP(F2E(f,LJ(c)), 0),
        (c<(k/2)?-1.:1.) * factor *C(f,LV(c)));
    }
  }
  L.resize(E.rows(),E.rows());
  L.setFromTriplets(LIJV.begin(),LIJV.end());
  */
  igl::crouzeix_raviart_cotmatrix(V, F, E, EMAP, L);

  igl::normal_derivative(V, F, N);

  Eigen::SparseMatrix<double> Ad(E.rows(),V.rows());
	{
    std::vector<Eigen::Triplet<double> > AIJV(E.size());
    for (int e = 0; e < E.rows(); e++)
    {
	    for(int c = 0;c<E.cols(); c++)
	    {
				// A triplet is a simple object representing a non-zero entry as the triplet: row index, column index, value. 
	      AIJV[e + c * E.rows()] = Eigen::Triplet<double>(e, E(e, c), 1);
	    }
	  }
	  Ad.setFromTriplets(AIJV.begin(),AIJV.end());
	}

  
  {
      std::vector<Eigen::Triplet<double>> ZIJV;
      for(int t =0;t<F.rows();t++)
      {
        for(int f =0;f<F.cols();f++)
        {
          if(C(t,f))
          {
            const int i = t+f*F.rows();
            for(int c = 1;c<F.cols();c++)
            {
              ZIJV.emplace_back(F(t,(f+c)%F.cols()),i,1);
            }
          }
        }
      }
      Z.resize(V.rows(),N.rows());
      Z.setFromTriplets(ZIJV.begin(),ZIJV.end());
      N = (Z*N).eval();
    }

  std::cout << "DEBUG: Normal derivative:\n" << N << std::endl;

  //  std::cout << "before degrees of freedom" << std::endl;

  Eigen::Matrix<double, Eigen::Dynamic, 1> De;

  igl::sum(Ad, 2, De);

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> De_diag =
	  De.array().inverse().matrix().asDiagonal();


  K = L * (De_diag * Ad);
  // K = L + N;

  std::cout << "DEBUG: Have k" << std::endl;

  // this is the wrong implementation
  igl::invert_diag(M, M_inv);


	// kill boundary edges, this is for closed surfaces 
	for(int f = 0;f<F.rows();f++)
	{
	  for(int c = 0;c<F.cols();c++)
	  {
			// this is a boundary edge so remove it 
	    if(C(f,c))
	    {
	      const int e = EMAP(f+F.rows()*c);
	      M_inv.diagonal()(e) = 0;
	    }
	  }
	}

  // radial basis functions

  Q = K.transpose() * (M_inv * K);


  CFile << C << std::endl;
  cotLFile << L << "\noriginal cot laplacian:\n" << cotLaplacian << std::endl;
  KFile << "K:\n" << K << "\nN:\n" << N << std::endl;
  massMatrixFile << M << "\nM_inv:\n"
                 << M_inv << "\noriginal mass matrix:\n" << massMatrix << std::endl;
  AdFile << Ad << "\n DOF:\n"
         << De << std::endl;
  AdFile << "\nDOF * Ad:\n" << (De_diag * Ad) << std::endl;

  igl::min_quad_with_fixed_precompute(Q, b, Eigen::SparseMatrix<double>(), true, data);

  CFile.close();
  cotLFile.close();
  massMatrixFile.close();
  AdFile.close();
  KFile.close();

#endif
  // A_eq is just the initialzied sparse matrix... but why?

  data.n = V.rows();
}
