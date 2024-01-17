#include <iostream>
using namespace std;

//int dem = 0;

const int MAX = 100;

int nhap[MAX][MAX]; //mang nhap, quyen nhap

int C(int m, int n) {
	if(m == 0 || n == 0)
		return nhap[m][n] = 1;
	
	//dem++;
	
	// neu ma nhap trang thi tinh toan va luu vao nhap	
	if(nhap[m][n] == -1)
		nhap[m][n] = C(m-1,n) + C(m, n-1);
	return nhap[m][n];
}
int main()
{
	for(int i=0; i < MAX; i++)
		for(int j = 0; j < MAX; j++)
			nhap[i][j] = -1; //xoa trang, danh dau nhap
	
	int m, n;
	cout << "Nhap M = "; cin >> m;
	cout << "Nhap N = "; cin >> n;
	
	cout << "So cach = " << C(m,n);
	
	//cout << "\nSo lan goi ham: " << dem;
	return 0;
}

