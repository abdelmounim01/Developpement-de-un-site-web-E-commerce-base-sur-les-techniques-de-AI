import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { ActivatedRoute } from '@angular/router';
import { Task } from 'src/app/service/Interfaces/task';
import { AuthService } from 'src/app/service/auth/auth.service';
import { PriceService } from 'src/app/service/PriceService/price.service';



@Component({
  selector: 'app-portfolio-images',
  templateUrl: './portfolio-images.component.html',
  styleUrls: ['./portfolio-images.component.css']
})
export class PortfolioImagesComponent implements OnInit {
  tasks: Task[];
  maylikes: Task[]
  populars: Task[]
  categorie : string ; 
  subCategorie: string ; 
  searchText;
  search: string;
  p: number=1;
  iduser:number = JSON.parse(localStorage.getItem('user')).userId;
collection = [];
message:string;


  slideConfig = {"slidesToShow": 5, "slidesToScroll": 4};
  constructor(private price: PriceService , private resourceService:ProjetService ,private route:ActivatedRoute,private authService: AuthService) {
    for (let i = 1; i <= 100; i++) {
      this.collection.push(`item ${i}`);
    }
  }


  ngOnInit(): void {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

   
   this.categorie = this.route.snapshot.paramMap.get('categorie') ; 
   this.subCategorie = this.route.snapshot.paramMap.get('subCategorie') ;
   if ((this.categorie) && !(this.subCategorie)) {
    this.maylikecat(this.iduser,this.categorie)
    this.getCategDyn(this.categorie )
    this.MostPopularcat (this.categorie)
    
   }
   else if((this.categorie) && (this.subCategorie)){
    this.maylikesubcat(this.iduser,this.categorie, this.subCategorie)
    
    this.getSubcategory(this.categorie , this.subCategorie)
    this.MostPopularsubcat (this.categorie,this.subCategorie)
    
   }
   else if(!(this.categorie) && !(this.subCategorie)){
    this.maylike(this.iduser)
    this.MostPopular()
    
   }
   
  }
 
  MostPopular (): void {
    this.resourceService.popularproducts().subscribe(populars => (this.populars = populars))
  }

  MostPopularcat (categorie): void {
    this.resourceService.popularproductscat(categorie).subscribe(populars => (this.populars = populars))
  }

  MostPopularsubcat (categorie,subCategorie): void {
    this.resourceService.popularproductssubcat(categorie,subCategorie).subscribe(populars => (this.populars = populars))
  }

  

  maylike(iduser):void {
    this.resourceService.maylike(this.iduser).subscribe(maylikes => (this.maylikes = maylikes))
  }

  maylikecat(iduser,categorie):void {
    this.resourceService.maylikecat(this.iduser,categorie).subscribe(maylikes => (this.maylikes = maylikes))
  }

  maylikesubcat(iduser,categorie,subCategorie):void {
    this.resourceService.maylikesubcat(this.iduser,categorie,subCategorie).subscribe(maylikes => (this.maylikes = maylikes))
  }

  getCategDyn(cat){
   this.resourceService.getInfoCateg(cat).subscribe(tasks =>{
      this.tasks = tasks
      
      })  
 }

 
 getSubcategory(cat , subCat){
  this.resourceService.getSubCategory(cat , subCat).subscribe(tasks =>{
     this.tasks = tasks
     })  
}


getLogin() {
  return this.authService.getUser().login;
}

logout() {
  return this.authService.logout();
}

  }
