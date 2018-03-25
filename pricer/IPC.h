#ifndef IPC_H
#define IPC_H

#include <sys/un.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stddef.h>
#include <errno.h>
#include <stdio.h>
#include <pthread.h>
#include <malloc.h>

class IPC
{
private:
  const char *name;
  int ipcListen();
  int ipcAccept(int ipc_fd, uid_t *uidptr);

public:
  IPC(const char *name);
  void start(void *(*routine)(void *));
};

#endif