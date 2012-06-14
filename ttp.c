#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

// void *mmap(void *addr, size_t length, int prot, int flags,
//                  int fd, off_t offset);

const int alpha_size = 46;
const int order = 4;

// typedef short int16;
typedef struct {
  char* start;
  int length;
} range;



#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

range* my_mmap (char* filename) {
    struct stat sb;

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        handle_error("open");

    if (fstat(fd, &sb) == -1)           /* To obtain file size */
        handle_error("fstat");

    char* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
        handle_error("mmap");
    range *r = (range*) malloc (sizeof(range));
    r->start = addr;
    r->length = sb.st_size;

    // (range { start = char, length = sb.st_size});
    return r;
}

int** getCorpus (char* filename) {
  range* r = my_mmap (filename);
  int** t = (int**) malloc(order * sizeof(int*));
  char* end = r->start + r->length;
  for (int i = 0, size=1; i < order; i++, size*=alpha_size) {
    t[i] = (int*) malloc(size * sizeof(int)); }
  for (char* s = r->start; s < end; s++) {
    for (int i = 0; (i < order) && (s+i < end); i++) {
      
    } }
  return t; }


int main () {
  // assert (sizeof(int16) >= 2);
  // printf("%s %i %i\n", bla+3, (bla+4) < (bla+s), s);
  getCorpus ("corpus");
  return 0;
}
