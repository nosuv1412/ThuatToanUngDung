#include <iostream>
#define MAX 100
using namespace std;

void NhapMang(int a[], int n)
{
	for(int i=0; i<n; i++)
	{
		cout<<"Nhap phan tu thu "<< i+1 <<": ";
		cin>>a[i];
	}
}
void XuatMang(int a[], int n)
{
	for(int i=0; i<n; i++)
		cout << a[i] <<" ";
}

float TinhTBC(int a[], int n){
	float tong = 0;
	for(int i = 0; i < n; i++){
		tong += a[i];
	}
	float tbc = tong/n;
	return tbc;
}

int DemSoAm(int a[], int n){
	int dem = 0;
	for(int i = 0; i < n; i++){
		if(a[i] < 0){
			dem++;
		}
	} return dem;
}

//int searchMin(int a[], int n){
//	int k = 0;
//	for (int i = 1; i < n; i++){
//		if (a[k] > a[i]){
//			k = i;
//		}
//	}
//	return k;
//}
//int searchMax(int a[], int n){
//	int k = 0;
//	for (int i = 1; i < n; i++){
//		if (a[k] <= a[i]){
//			k = i;
//		}
//	}
//	return k;
//}

int main()
{
	int a[MAX], n;
	cout << "Nhap N phan tu: "; cin >> n;
	NhapMang(a,n);
	cout<< "Cac phan tu trong mang: ";
	XuatMang(a,n);
	cout << "\nTBC cua cac phan tu trong day la: " << TinhTBC(a,n);
	cout << "\nSo am co trong mang: " << DemSoAm(a,n) << " so";
	
//	int so_Min = searchMin(a,n);
//	int so_Max = searchMax(a,n);
//	swap(a[so_Min], a[so_Max]);
//	cout<< "\nDay so sau khi doi cho la: ";
//	XuatMang(a,n);	
return 0;
}

