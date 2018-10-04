#include <bits/stdc++.h>
#include <random>
using namespace std;       
#define fr(i,n)     for(int i=0;i<n;i++)
#define frr(i,a,b)   for(int i=a;i<=b;i++) 
#define rrf(i,a,b)   for(int i=a;i>=b;i--)
#include <string>
#include <sstream> 
#include <stdlib.h>
#include <stdio.h>   
#include <algorithm>
#define inf 2000000000
#include <fstream>
#define vc(r) vector < r >

bool valid(int i,int j,int h,int w)
{
    return (i>=0 && i<h && j>=0 && j<w);
}

std::string tostring(int i)
{
    std::stringstream ss;
    ss << i;
    return ss.str();
}

int main ()
{
    cout<<"Please enter the file number"<<endl;
    fstream f,g;
    int n;cin>>n; // file number
    cout<<"Please enter the desired ratio of new number of 1's to original (in decimals greater than 1)"<<endl;
    float ratio;cin>>ratio; // i will assume that ration is more than 1

    string sec=".txt";
    string mid=tostring(n);
    string filename=mid+sec;
    string filename2="f"+mid+sec;

    f.open(filename.c_str());
    const int h=60,w=60;
    int data[h][w][4];
    
    vc( vc(int) )vector_of_ones;
    int ones=0;

    fr(k,h*w)
    {
        int i=k/h;
        int j=k%h;
        f>>data[i][j][0]>>data[i][j][1]>>data[i][j][2];
        f>>data[i][j][3];
        ones+=data[i][j][3];
    }
    

    std::random_device rd;
    std::mt19937 g1(rd());
    std::shuffle(std::begin(data),std::end(data),g1);

    // data has been shuffled

    vector_of_ones.resize(ones);
    int index=0;
    
    // yes , data now is stored correctly

    f.close();
    g.open(filename2.c_str(), std::fstream::out);

    cout<<ones<<endl;
    fr(k,h*w)
    {
        int i=k/h;
        int j=k%h;
        // ======================= normal orientation
        frr(a,-2,2)
        {
            frr(b,-2,2)
            {
                if(valid(i+a,j+b,h,w))
                  		g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";              			
                else	
                    g<<"50 150 100 ";    		
            }
        }
        g<<data[i][j][3]<<endl;
        // =======================================
        frr(a,-2,2)
        {
            rrf(b,2,-2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //===========================================
        rrf(a,2,-2)
        {
            frr(b,-2,2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //============================================
        rrf(a,2,-2)
        {
            rrf(b,2,-2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //=============================================

         frr(b,-2,2)
        {
            frr(a,-2,2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        // =======================================
        frr(b,-2,2)
        {
            rrf(a,2,-2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //===========================================
        rrf(b,2,-2)
        {
            frr(a,-2,2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //============================================
        rrf(b,2,-2)
        {
            rrf(a,2,-2)
            {
                if(valid(i+a,j+b,h,w))
                    g<<data[i+a][j+b][0]<<" "<<data[i+a][j+b][1]<<" "<<data[i+a][j+b][2]<<" ";
                else
                    g<<"50 150 100 ";
            }
        }
        g<<data[i][j][3]<<endl;
        //=============================================

        if(data[i][j][3]==1)
        {
        	frr(a,-2,2)
        	{
        		frr(b,-2,2)
        		{
        			if(valid(i+a,j+b,h,w))
        			{
        				vector_of_ones[index].push_back(data[i+a][j+b][0]);
        				vector_of_ones[index].push_back(data[i+a][j+b][1]);
        				vector_of_ones[index].push_back(data[i+a][j+b][2]);	
        			}
        			else
        			{
        				vector_of_ones[index].push_back(50);
        				vector_of_ones[index].push_back(150);
        				vector_of_ones[index].push_back(100);
        			}
        		}
        	}
        	index++;
        }
    }

     // all orientations are now done ..... its time to handle the ratio

    int full = (int)ratio;
    float fraction = (ratio-full);
    int part = fraction*ones;

    fr(n,full)
    {
    	fr(i,ones)
    	{
    		fr(j,75)g<<vector_of_ones[i][j]<<" ";
    		g<<"1"<<endl;
    	}
    }

    fr(i,part)
    {
    	fr(j,75)g<<vector_of_ones[i][j]<<" ";
    	g<<"1"<<endl;
    }
    return 0;
}

