#include <iostream>
using namespace std;
const int MAX = 20;
int n , a[MAX];
void print(int x){
	cout << n << " = " << a[1];
	for(int i = 2 ; i<= x ; i++){
		cout << "+" << a[i];
	}
	cout<<endl;
}
int checkHoanVi(int k)
{
	for (int i=1 ; i<k ; i++)
		if (a[i] < a[i+1])
			return 0;
	return 1;
}
void gen(int k, int q)
{
	if(q == n && checkHoanVi(k))
	{
		print(k - 1);
		return;
	}
	for(int i = n-q ; i >=1 ; i--){
		a[k] = i;
		gen(k+1, q+i);
	}
}
int main()
{
	cout<<"Nhap n = ";
	cin >> n;
	gen(1,0);
}

