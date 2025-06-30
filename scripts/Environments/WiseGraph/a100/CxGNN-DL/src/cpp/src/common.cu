#include "common.h"
#include "spdlog/spdlog.h"
#include "sys/sysinfo.h"
#include "sys/types.h"

double cudaMemInfo()  // return MB
{
  cudaDeviceSynchronize();
  size_t free_byte;
  size_t total_byte;
  auto cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if (cudaSuccess != cuda_status) {
    printf("Error: cudaMemGetInfo fails, %s \n",
           cudaGetErrorString(cuda_status));
    exit(1);
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  // printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
  //        used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db /
  //        1024.0 / 1024.0);
  return used_db / 1024.0 / 1024.0;
}

void showCpuMem() {
  struct sysinfo memInfo;
  sysinfo(&memInfo);
  long long totalVirtualMem = memInfo.totalram;
  // Add other values in next statement to avoid int overflow on right hand
  // side...
  totalVirtualMem += memInfo.totalswap;
  totalVirtualMem *= memInfo.mem_unit;
  long long virtualMemUsed = memInfo.totalram - memInfo.freeram;
  // Add other values in next statement to avoid int overflow on right hand
  // side...
  virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
  virtualMemUsed *= memInfo.mem_unit;
  long long totalPhysMem = memInfo.totalram;
  // Multiply in next statement to avoid int overflow on right hand side...
  totalPhysMem *= memInfo.mem_unit;
  long long physMemUsed = memInfo.totalram - memInfo.freeram;
  // Multiply in next statement to avoid int overflow on right hand side...
  physMemUsed *= memInfo.mem_unit;
  SPDLOG_WARN(
      "total-virtual-mem {} used-virtual-mem {} total-phy-mem {} used-phy-mem "
      "{}",
      totalVirtualMem, virtualMemUsed, totalPhysMem, physMemUsed);
}

void showCpuMemCurrProc() {
  double vm_usage, resident_set;
  using std::ifstream;
  using std::ios_base;
  using std::string;

  vm_usage = 0.0;
  resident_set = 0.0;

  // 'file' stat seems to give the most reliable results
  //
  ifstream stat_stream("/proc/self/stat", ios_base::in);

  // dummy vars for leading entries in stat that we don't care about
  //
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  string O, itrealvalue, starttime;

  // the two fields we want
  //
  unsigned long vsize;
  long rss;

  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr >>
      tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >> utime >>
      stime >> cutime >> cstime >> priority >> nice >> O >> itrealvalue >>
      starttime >> vsize >> rss;  // don't care about the rest

  stat_stream.close();

  long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                      1024;  // in case x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;
  SPDLOG_WARN("Curr-proc VM {} Res {}", vm_usage, resident_set);
}

double getAverageTimeWithWarmUp(const std::function<void()> &f) {
  const int nWarmUps = 3;
  const int nRuns = 3;
  for (int i = 0; i < nWarmUps; ++i) f();
  double totalTime = 0;
  for (int i = 0; i < nRuns; ++i) {
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t0);
    f();
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t1);
    totalTime += getDuration(t0, t1);
  }
  return totalTime / nRuns;
}
