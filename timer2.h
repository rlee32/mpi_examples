#include <sys/time.h>

static struct timeval timer_start, timer_end;

void start_timer()
{
  gettimeofday(&timer_start, NULL);
}

double stop_timer()
{
  gettimeofday(&timer_end, NULL);
  double delta = ((timer_end.tv_sec  - timer_start.tv_sec) * 1000000u + 
    timer_end.tv_usec - timer_start.tv_usec) / 1.e6;
  return delta;
}