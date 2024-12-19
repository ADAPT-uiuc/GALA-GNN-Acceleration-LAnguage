#ifndef _LIKWID_DEFS_H
#define _LIKWID_DEFS_H

#ifdef LIKWID_PERFMON
#include <likwid.h>
#include <omp.h>
#include <string>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif


void likwid_init(){
#ifdef LIKWID_PERFMON 
    LIKWID_MARKER_INIT;
    #pragma omp parallel
    {
        LIKWID_MARKER_THREADINIT;
    }
#endif
}

void likwid_named_marker_init(std::string name){
#ifdef LIKWID_PERFMON 
    #pragma omp parallel
    {
        LIKWID_MARKER_START(name.c_str());
        LIKWID_MARKER_STOP(name.c_str());
        
    }
#endif
}

void likwid_marker_start_parallel(std::string name){
#ifdef LIKWID_PERFMON 
    #pragma omp parallel
    {
        LIKWID_MARKER_START(name.c_str());
        // LIKWID_MARKER_STOP(name.c_str());
        
    }
#endif
}

void likwid_marker_stop_parallel(std::string name){
#ifdef LIKWID_PERFMON 
    #pragma omp parallel
    {
        // LIKWID_MARKER_START(name.c_str());
        LIKWID_MARKER_STOP(name.c_str());
        
    }
#endif
}

void likwid_close(){
#ifdef LIKWID_PERFMON 
    LIKWID_MARKER_CLOSE;
#endif
}

// per thread
void likwid_marker_start(std::string name){
#ifdef LIKWID_PERFMON 
    LIKWID_MARKER_START(name.c_str());    
#endif   
}

void likwid_marker_stop(std::string name){
#ifdef LIKWID_PERFMON 
    LIKWID_MARKER_STOP(name.c_str());    
#endif   
}
#endif