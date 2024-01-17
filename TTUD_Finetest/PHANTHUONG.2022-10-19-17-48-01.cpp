#include<iostream>
#include<vector>
using namespace std;
 
struct canh {
    int a, b, t;
};
 
int n, m, vc = 1000000000, duongdi[1001], V[1001];
 
long long phanthuong[10001];
 
vector <canh> E;
 
void khoitao() {
    duongdi[1] = 0;
    phanthuong[1] = V[1];
    for (int i = 2; i <= n; i++) {
        duongdi[i] = vc;
        phanthuong[i] = 0;
    }
}
 
void dichuyen() {
    for (int e = 0; e < E.size(); e++)
        for (int i = 0; i < E.size(); i++)
            if (duongdi[E[i].a] != vc) {
                if (duongdi[E[i].b] > (duongdi[E[i].a] + E[i].t))
                    phanthuong[E[i].b] = phanthuong[E[i].a] + V[E[i].b];
                if (duongdi[E[i].b] == (duongdi[E[i].a] + E[i].t))
                    phanthuong[E[i].b] = max(phanthuong[E[i].b], (phanthuong[E[i].a] + V[E[i].b]));
                duongdi[E[i].b] = min(duongdi[E[i].b], duongdi[E[i].a] + E[i].t);
            }
}
 
int main() {
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> V[i];
    cin >> m;
    for (int i = 0; i < m; i++) {
        int p, q, t;
        cin >> p >> q >> t;
        canh temp = { p, q, t };
        E.push_back(temp);
        temp = { q, p, t };
        E.push_back(temp);
    }
 
    khoitao();
    dichuyen();
    if (duongdi[n] == vc) cout << 0;
    else cout << phanthuong[n];
}
