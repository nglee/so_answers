struct stopwatch_t {
    double time;
};
void stopwatch_init();
struct stopwatch_t *stopwatch_create();
void stopwatch_start(struct stopwatch_t *timer);
long double stopwatch_stop(struct stopwatch_t *timer);
void stopwatch_destroy(struct stopwatch_t *timer);
