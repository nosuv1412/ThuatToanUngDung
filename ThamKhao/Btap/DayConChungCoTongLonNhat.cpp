#include <iostream>
using namespace std;

const int MAX = 1000;
int a[MAX], b[MAX], c[MAX][MAX], m, n;

void sumMax() {
	for(int i = 0; i <= m; i++) {
		for(int j = 0; j <= n; j++) {
			if(i == 0 || j == 0) {
				c[i][j] = 0;
			} else {
				if(a[i] == b[j]) {
					c[i][j] = c[i-1][j-1] + a[i];
				} else {
					c[i][j] = max(c[i-1][j], c[i][j-1]);
				}
			}
		}
	}
}

/*
	a = [3, 4, 1, 1];
	b = [3, 1, 1, 4];
	i=0, j=0: c[0,0] = 0;
	i=0, j=1: c[0,1] = 0;
	i=0, j=2: c[0,2] = 0;
	i=0, j=3: c[0,3] = 0;
	i=0, j=4: c[0,4] = 0;
	
	i=1, j=0: c[1,0] = 0;
	i=1, j=1: c[1,1] = c[0,0] + 3 = 3;
	i=1, j=2: c[1,2] = c[1,1] = 3;
	i=1, j=3: c[1,3] = c[1,2] = 3;
	i=1, j=4: c[1,4] = c[1,4] = 3;
	
	i=2, j=0: c[2,0] = c[0,0] + 3 = 3;

*/

int main() {
	cout << "Nhap m = "; cin >> m;
	for(int i = 1; i <= m; i++) {
		cout<<"a["<<i<<"] = "; cin>>a[i];
	}
	
	cout << "Nhap n = "; cin >> n;
	for(int i = 1; i <= n; i++) {
		cout<<"b["<<i<<"] = "; cin>>b[i];
	}
	
	sumMax();
	
	cout<<"Day con co tong lon nhat = "<<c[m][n];
}
