#include <iostream>

using namespace std;

int main()
{
 	int m, n;
    cout << "Nhap M = ";
    cin >> m;
    cout << "Nhap N = ";
    cin >> n;
    int a[m+1][n+1];
    for(int i=0;i<=m;i++){
    	for(int j=0;j<=n;j++){
    		a[i][j] = (i==0 || j == 0) ? 1 : a[i-1][j] + a[i][j-1];
		}
	}
	cout<<"So cach = "<<a[m][n];
return 0;
}

