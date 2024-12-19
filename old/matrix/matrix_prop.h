#ifndef _MATRIX_PROP_H
#define _MATRIX_PROP_H
#include <string>
enum class MtxStructure { HAS_L, HAS_U, HAS_D, HAS_LD, HAS_DU, HAS_LDU };

enum class MtxSymmetry { SYMMETRIC, HERMITIAN, GENERAL, UNKNOWN };

struct MatrixProperties {
  MtxStructure structure_;
  MtxSymmetry symmetry_;

  MatrixProperties() : structure_(MtxStructure::HAS_LDU), symmetry_(MtxSymmetry::GENERAL) {}
};

enum class CSRC_TYPE {
    CSR, CSC, HCSR, HCSC, COO_CO, COO_RO, DCSR
};

struct DistStats{
  double min_;
  double min_nz_;
  double max_;
  double max_nz_;
  double sum_;
  double mean_;
  double mean_nz_;
  double median_;
  double median_nz_;
  double variance_;
  double variance_nz_;
  double stddev_;
  double stddev_nz_;
  double gini_;
  double gini_nz_;
  double pratio_;
  double pratio_nz_;
  double nonzero_;
  double nelems_;
  double nonzero_ratio_;
};

struct MtxStats {
  std::string mtx_name;
  int target_dim_;
  double blk_dimr_;
  double blk_dimc_;
  double mtx_dimr_;
  double mtx_dimc_;
  // #of rows/columns (vertices)
  double nrows_, ncols_;
  // #of non-zeros (edges)
  double nvals_;
  // Mean. #nnz per row/col
  double r_mean_, c_mean_;
  double rb_mean_, cb_mean_, t_mean_;

  double r_mean_ne_, c_mean_ne_;
  double rb_mean_ne_, cb_mean_ne_, t_mean_ne_;

  // Std. dev #nnz per row/col
  double r_stddev_, c_stddev_;
  double rb_stddev_, cb_stddev_, t_stddev_;

  // Variance #nnz per row/col
  double r_var_, c_var_;
  double rb_var_, cb_var_, t_var_;

  //   min max
  double r_min_, r_max_;
  double c_min_, c_max_;
  double rb_min_, rb_max_;
  double cb_min_, cb_max_;
  double t_min_, t_max_;
  // Gini index
  double r_gini_, c_gini_;
  double rb_gini_, cb_gini_, t_gini_;

  double r_nonempty_, c_nonempty_;
  double r_nonempty_ratio_, c_nonempty_ratio_;
  double rb_nonempty_, cb_nonempty_, t_nonempty_;
  double rb_nonempty_ratio_, cb_nonempty_ratio_, t_nonempty_ratio_;

  // P-ratio
  double r_pratio_, c_pratio_;
  double rb_pratio_, cb_pratio_, t_pratio_;

  //   locality stats
  double ratio_uniqc_, ratio_uniqr_;
  double ratio_uniqc4_, ratio_uniqr4_;
  double ratio_uniqc8_, ratio_uniqr8_;
  double ratio_uniqc16_, ratio_uniqr16_;
  double ratio_uniqc32_, ratio_uniqr32_;
  double ratio_uniqc64_, ratio_uniqr64_;
  double ptL3reuser_, ptL3reuser4_, ptL3reuser8_, ptL3reuser16_, ptL3reuser32_, ptL3reuser64_;
  double ptL3reusec_, ptL3reusec4_, ptL3reusec8_, ptL3reusec16_, ptL3reusec32_, ptL3reusec64_;
  // double blk_dimr, blk_dimc;

  //   tile stats
  double ratio_nonempty_tiles;
  double nonempty_tiles;
  //   double ratio_nonempty_tiles;
  //   double nonempty_tiles;
  void set_mtx_name(std::string mtx_namex){
    mtx_name = mtx_namex;
  }
  void set_generic_mtx_prop(double nrows, double ncols, double nvals, int target_dim, double blk_dimr,
    double blk_dimc, double mtx_dimr, double mtx_dimc){
      nrows_ = nrows;
      ncols_ = ncols;
      nvals_ = nvals;
      target_dim_ = target_dim;
      blk_dimr_ = blk_dimr;
      blk_dimc_ = blk_dimc;
      mtx_dimr_ = mtx_dimr;
      mtx_dimc_ = mtx_dimc;
      
    }
  void set_row_stats(DistStats row_stats){
    r_mean_ = row_stats.mean_;
    r_stddev_ = row_stats.stddev_;
    r_var_ = row_stats.variance_;
    r_min_ = row_stats.min_;
    r_max_ = row_stats.max_;
    r_nonempty_ = row_stats.nonzero_;
    r_nonempty_ratio_ = row_stats.nonzero_ratio_;
    r_gini_ = row_stats.gini_;
    r_pratio_ = row_stats.pratio_;
  }
  void set_col_stats(DistStats col_stats){
    c_mean_ = col_stats.mean_;
    c_stddev_ = col_stats.stddev_;
    c_var_ = col_stats.variance_;
    c_min_ = col_stats.min_;
    c_max_ = col_stats.max_;
    c_nonempty_ = col_stats.nonzero_;
    c_nonempty_ratio_ = col_stats.nonzero_ratio_;
    c_gini_ = col_stats.gini_;
    c_pratio_ = col_stats.pratio_;
  }
  void set_rb_stats(DistStats rb_stats){
// std::cout<<"RowBlk==> mean: "<<rb_mean<<" stddev: "<<rb_stddev<<" var: "<<rb_var<<" min: "<<rb_min<<" max: "<<rb_max<<" nonempty: "<<rb_nonempty<<" nonempty_r: "<<rb_nonempty_ratio<<" gini: "<<rb_gini<<" pratio: "<<rb_pratio<<std::endl;
    rb_mean_ = rb_stats.mean_;
    rb_stddev_ = rb_stats.stddev_;
    rb_var_ = rb_stats.variance_;
    rb_min_ = rb_stats.min_;
    rb_max_ = rb_stats.max_;
    rb_nonempty_ = rb_stats.nonzero_;
    rb_nonempty_ratio_ = rb_stats.nonzero_ratio_;
    rb_gini_ = rb_stats.gini_;
    rb_pratio_ = rb_stats.pratio_;
  }
  void set_cb_stats(DistStats cb_stats){
// std::cout<<"ColBlk==> mean: "<<cb_mean<<" stddev: "<<cb_stddev<<" var: "<<cb_var<<" min: "<<cb_min<<" max: "<<cb_max<<" nonempty: "<<cb_nonempty<<" nonempty_r: "<<cb_nonempty_ratio<<" gini: "<<cb_gini<<" pratio: "<<cb_pratio<<std::endl;
    cb_mean_ = cb_stats.mean_;
    cb_stddev_ = cb_stats.stddev_;
    cb_var_ = cb_stats.variance_;
    cb_min_ = cb_stats.min_;
    cb_max_ = cb_stats.max_;
    cb_nonempty_ = cb_stats.nonzero_;
    cb_nonempty_ratio_ = cb_stats.nonzero_ratio_;
    cb_gini_ = cb_stats.gini_;
    cb_pratio_ = cb_stats.pratio_;    
  }
  void set_tile_stats(DistStats tile_stats){
//  std::cout<<"Blk==> mean: "<<t_mean<<" stddev: "<<t_stddev<<" var: "<<t_var<<" min: "<<t_min<<" max: "<<t_max<<" nonempty: "<<t_nonempty<<" nonempty_r: "<<t_nonempty_ratio<<" gini: "<<t_gini<<" pratio: "<<t_pratio<<std::endl;
    t_mean_ = tile_stats.mean_;
    t_stddev_ = tile_stats.stddev_;
    t_var_ = tile_stats.variance_;
    t_min_ = tile_stats.min_;
    t_max_ = tile_stats.max_;
    t_nonempty_ = tile_stats.nonzero_;
    t_nonempty_ratio_ = tile_stats.nonzero_ratio_;
    t_gini_ = tile_stats.gini_;
    t_pratio_ = tile_stats.pratio_;  
  }

  void set_col_locality(double K, double uniq, double uniq4, double uniq8, double uniq16, double uniq32, double uniq64){
    ratio_uniqc_ = (double)(uniq) / (double)(nvals_);
    ptL3reusec_ = (double)(uniq) / (double)(ncols_);
    ratio_uniqc4_ = (double)(uniq4) / (double)(nvals_);
    ptL3reusec4_ = (double)(uniq4) / (double)(ncols_/4.0);
    ratio_uniqc8_ = (double)(uniq8) / (double)(nvals_);
    ptL3reusec8_ = (double)(uniq8) / (double)(ncols_/8.0);
    ratio_uniqc16_ = (double)(uniq16) / (double)(nvals_);
    ptL3reusec16_ = (double)(uniq16) / (double)(ncols_/16.0);
    ratio_uniqc32_ = (double)(uniq32) / (double)(nvals_);
    ptL3reusec32_ = (double)(uniq32) / (double)(ncols_/32.0);
    ratio_uniqc64_ = (double)(uniq64) / (double)(nvals_);
    ptL3reusec64_ = (double)(uniq64) / (double)(ncols_/64.0);
  }

  void set_row_locality(double K, double uniq, double uniq4, double uniq8, double uniq16, double uniq32, double uniq64){
    ratio_uniqr_ = (double)(uniq) / (double)(nvals_);
    ptL3reuser_ = (double)(uniq) / (double)(nrows_/1.0);
    ratio_uniqr4_ = (double)(uniq4) / (double)(nvals_);
    ptL3reuser4_ = (double)(uniq4) / (double)(nrows_/4.0);
    ratio_uniqr8_ = (double)(uniq8) / (double)(nvals_);
    ptL3reuser8_ = (double)(uniq8) / (double)(nrows_/8.0);
    ratio_uniqr16_ = (double)(uniq16) / (double)(nvals_);
    ptL3reuser16_ = (double)(uniq16) / (double)(nrows_/16.0);
    ratio_uniqr32_ = (double)(uniq32) / (double)(nvals_);
    ptL3reuser32_ = (double)(uniq32) / (double)(nrows_/32.0);
    ratio_uniqr64_ = (double)(uniq64) / (double)(nvals_);
    ptL3reuser64_ = (double)(uniq64) / (double)(nrows_/64.0);
  }

  void print(){
      std::cout<<"For humans:"<<std::endl;
      std::cout<<"Mtx: "<<mtx_name<<" target_dim: "<<target_dim_<<std::endl;
      std::cout<<"Mtx: nrows: "<< nrows_ << " ncols: "<<ncols_<<" nvals: "<<nvals_<<" blkdimr: "<<blk_dimr_<<" blkdimc: "<<blk_dimc_<<" mtxdimr: "<<mtx_dimr_<<" mtxdimc: "<<mtx_dimc_<<std::endl;
      std::cout<<"Row==> mean: "<<r_mean_<<" stddev: "<<r_stddev_<<" var: "<<r_var_<<" min: "<<r_min_<<" max: "<<r_max_<<" nonempty: "<<r_nonempty_<<" nonempty_r: "<<r_nonempty_ratio_<<" gini: "<<r_gini_<<" pratio: "<<r_pratio_<<std::endl;
      std::cout<<"Col==> mean: "<<c_mean_<<" stddev: "<<c_stddev_<<" var: "<<c_var_<<" min: "<<c_min_<<" max: "<<c_max_<<" nonempty: "<<c_nonempty_<<" nonempty_r: "<<c_nonempty_ratio_<<" gini: "<<c_gini_<<" pratio: "<<c_pratio_<<std::endl;
      std::cout<<"Blk==> mean: "<<t_mean_<<" stddev: "<<t_stddev_<<" var: "<<t_var_<<" min: "<<t_min_<<" max: "<<t_max_<<" nonempty: "<<t_nonempty_<<" nonempty_r: "<<t_nonempty_ratio_<<" gini: "<<t_gini_<<" pratio: "<<t_pratio_<<std::endl;
      std::cout<<"RowBlk==> mean: "<<rb_mean_<<" stddev: "<<rb_stddev_<<" var: "<<rb_var_<<" min: "<<rb_min_<<" max: "<<rb_max_<<" nonempty: "<<rb_nonempty_<<" nonempty_r: "<<rb_nonempty_ratio_<<" gini: "<<rb_gini_<<" pratio: "<<rb_pratio_<<std::endl;
      std::cout<<"ColBlk==> mean: "<<cb_mean_<<" stddev: "<<cb_stddev_<<" var: "<<cb_var_<<" min: "<<cb_min_<<" max: "<<cb_max_<<" nonempty: "<<cb_nonempty_<<" nonempty_r: "<<cb_nonempty_ratio_<<" gini: "<<cb_gini_<<" pratio: "<<cb_pratio_<<std::endl;
      std::cout<<"CL=1==> ratio_uniqc: "<< ratio_uniqc_ << " ratio_uniqr: "<< ratio_uniqr_ <<std::endl;
      std::cout<<"CL=4==> ratio_uniqc: "<< ratio_uniqc4_ << " ratio_uniqr: "<< ratio_uniqr4_ <<std::endl;
      std::cout<<"CL=8==> ratio_uniqc: "<< ratio_uniqc8_ << " ratio_uniqr: "<< ratio_uniqr8_ <<std::endl;
      std::cout<<"CL=16==> ratio_uniqc: "<< ratio_uniqc16_ << " ratio_uniqr: "<< ratio_uniqr16_ <<std::endl;
      std::cout<<"CL=32==> ratio_uniqc: "<< ratio_uniqc32_ << " ratio_uniqr: "<< ratio_uniqr32_ <<std::endl;
      std::cout<<"CL=64==> ratio_uniqc: "<< ratio_uniqc64_ << " ratio_uniqr: "<< ratio_uniqr64_ <<std::endl;

      std::cout<<"CL=1==> ptL3reusec: "<< ptL3reusec_ << " ptL3reuser: "<< ptL3reuser_ <<std::endl;
      std::cout<<"CL=4==> ptL3reusec: "<< ptL3reusec4_ << " ptL3reuser: "<< ptL3reuser4_ <<std::endl;
      std::cout<<"CL=8==> ptL3reusec: "<< ptL3reusec8_ << " ptL3reuser: "<< ptL3reuser8_ <<std::endl;
      std::cout<<"CL=16==> ptL3reusec: "<< ptL3reusec16_ << " ptL3reuser: "<< ptL3reuser16_ <<std::endl;
      std::cout<<"CL=32==> ptL3reusec: "<< ptL3reusec32_ << " ptL3reuser: "<< ptL3reuser32_ <<std::endl;
      std::cout<<"CL=64==> ptL3reusec: "<< ptL3reusec64_ << " ptL3reuser: "<< ptL3reuser64_ <<std::endl;



      std::cout<<"Stats-"<<target_dim_<<","<<mtx_name<<",";
      std::cout<<""<< nrows_ << ","<<ncols_<<","<<nvals_<<","<<blk_dimr_<<","<<blk_dimc_<<","<<mtx_dimr_<<","<<mtx_dimc_<<",";
      std::cout<<""<<r_mean_<<","<<r_stddev_<<","<<r_var_<<","<<r_min_<<","<<r_max_<<","<<r_nonempty_<<","<<r_nonempty_ratio_<<","<<r_gini_<<","<<r_pratio_<<",";
      std::cout<<""<<c_mean_<<","<<c_stddev_<<","<<c_var_<<","<<c_min_<<","<<c_max_<<","<<c_nonempty_<<","<<c_nonempty_ratio_<<","<<c_gini_<<","<<c_pratio_<<",";
      std::cout<<""<<t_mean_<<","<<t_stddev_<<","<<t_var_<<","<<t_min_<<","<<t_max_<<","<<t_nonempty_<<","<<t_nonempty_ratio_<<","<<t_gini_<<","<<t_pratio_<<",";
      std::cout<<""<<rb_mean_<<","<<rb_stddev_<<","<<rb_var_<<","<<rb_min_<<","<<rb_max_<<","<<rb_nonempty_<<","<<rb_nonempty_ratio_<<","<<rb_gini_<<","<<rb_pratio_<<",";
      std::cout<<""<<cb_mean_<<","<<cb_stddev_<<","<<cb_var_<<","<<cb_min_<<","<<cb_max_<<","<<cb_nonempty_<<","<<cb_nonempty_ratio_<<","<<cb_gini_<<","<<cb_pratio_<<",";
      std::cout<<""<< ratio_uniqc_ << ","<< ratio_uniqr_ <<",";
      std::cout<<""<< ratio_uniqc4_ << ","<< ratio_uniqr4_ <<",";
      std::cout<<""<< ratio_uniqc8_ << ","<< ratio_uniqr8_ <<",";
      std::cout<<""<< ratio_uniqc16_ << ","<< ratio_uniqr16_ <<",";
      std::cout<<""<< ratio_uniqc32_ << ","<< ratio_uniqr32_ <<",";
      std::cout<<""<< ratio_uniqc64_ << ","<< ratio_uniqr64_ <<",";

      std::cout<<""<< ptL3reusec_ << ","<< ptL3reuser_ <<",";
      std::cout<<""<< ptL3reusec4_ << ","<< ptL3reuser4_ <<",";
      std::cout<<""<< ptL3reusec8_ << ","<< ptL3reuser8_ <<",";
      std::cout<<""<< ptL3reusec16_ << ","<< ptL3reuser16_ <<",";
      std::cout<<""<< ptL3reusec32_ << ","<< ptL3reuser32_ <<",";
      std::cout<<""<< ptL3reusec64_ << ","<< ptL3reuser64_ <<std::endl;;

  // double ratio_uniqc4, ratio_uniqr4;
  // double ratio_uniqc8, ratio_uniqr8;
  // double ratio_uniqc16, ratio_uniqr16;
  // double ratio_uniqc32, ratio_uniqr32;
  // double ratio_uniqc64, ratio_uniqr64;

      
  }
};

#endif