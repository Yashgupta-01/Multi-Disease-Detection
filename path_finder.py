#!/usr/bin/env python3
import sys
import heapq

def solve():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    grid = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    sx = int(next(it)) - 1
    sy = int(next(it)) - 1
    dx = int(next(it)) - 1
    dy = int(next(it)) - 1
    k = int(next(it))
    blocked = set()
    for _ in range(k):
        bx = int(next(it)) - 1
        by = int(next(it)) - 1
        blocked.add((bx, by))
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    INF = 10**18
    dist = [[INF] * m for _ in range(n)]
    dist[sx][sy] = 0
    pq = [(0, sx, sy)]
    while pq:
        cost, x, y = heapq.heappop(pq)
        if cost > dist[x][y]: continue
        max_val = -1
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in blocked:
                max_val = max(max_val, grid[nx][ny])
        if max_val == -1: continue
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < m and (nx, ny) not in blocked:
                add = 0
                if grid[nx][ny] < max_val:
                    add = max_val + 1 - grid[nx][ny]
                new_cost = cost + add
                if new_cost < dist[nx][ny]:
                    dist[nx][ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
    ans = dist[dx][dy]
    print(ans if ans < INF else -1)

solve()
