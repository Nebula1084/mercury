#include <IPC.h>

// the max connection number of the server
#define MAX_CONNECTION_NUMBER 5

IPC::IPC(const char *name) : name(name)
{
}

int IPC::ipcListen()
{
    int fd;
    struct sockaddr_un un;

    if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0)
        return -1;

    int len, rval;
    unlink(name);
    memset(&un, 0, sizeof(un));
    un.sun_family = AF_UNIX;
    strcpy(un.sun_path, name);
    len = offsetof(struct sockaddr_un, sun_path) + strlen(name);

    /* bind the name to the descriptor */
    if (bind(fd, (struct sockaddr *)&un, len) < 0)
        rval = -2;
    else
    {
        if (listen(fd, MAX_CONNECTION_NUMBER) < 0)
            rval = -3;
        else
            return fd;
    }
    int err;
    err = errno;
    close(fd);
    errno = err;
    return rval;
}

int IPC::ipcAccept(int ipc_fd, uid_t *uidptr)
{
    int client_fd, len, rval;
    time_t stale_time;
    struct sockaddr_un un;
    struct stat stat_buf;
    len = sizeof(un);

    if ((client_fd = accept(ipc_fd, (struct sockaddr *)&un, (socklen_t *)&len)) < 0)
        return -1;

    /* obtain the client's uid from its calling address */
    len -= offsetof(struct sockaddr_un, sun_path); /* len of pathname */
    un.sun_path[len] = 0;                          /* null terminate */
    if (stat(un.sun_path, &stat_buf) < 0)
        rval = -2;
    else
    {
        if (S_ISSOCK(stat_buf.st_mode))
        {
            if (uidptr != NULL)
                *uidptr = stat_buf.st_uid; /* return uid of caller */
            unlink(un.sun_path);           /* we're done with pathname now */
            return client_fd;
        }
        else
            rval = -3; /* not a socket */
    }

    int err;
    err = errno;
    close(client_fd);
    errno = err;
    return rval;
}

void IPC::start(void *(*routine)(void *))
{
    int ipc_fd, conn_fd;
    ipc_fd = this->ipcListen();
    if (ipc_fd < 0)
    {
        printf("Error[%d] when listening:%s\n", errno, strerror(errno));
        return;
    }

    printf("Finished listening...\n");
    uid_t uid;
    pthread_t thread;
    while (1)
    {
        conn_fd = this->ipcAccept(ipc_fd, &uid);

        if (conn_fd < 0)
        {
            printf("Error[%d] when accepting:%s\n", errno, strerror(errno));
            break;
        }

        int *fd = (int *)malloc(sizeof(int));
        *fd = conn_fd;
        pthread_create(&thread, NULL, routine, (void *)fd);
        pthread_detach(thread);
    }

    close(ipc_fd);
    printf("Server exited.\n");
}