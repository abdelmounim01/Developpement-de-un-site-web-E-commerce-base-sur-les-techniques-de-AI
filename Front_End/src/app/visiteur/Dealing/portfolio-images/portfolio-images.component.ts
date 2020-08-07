import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { ActivatedRoute } from '@angular/router';
import { Task } from 'src/app/service/Interfaces/task';
import { PriceService } from 'src/app/service/PriceService/price.service';




@Component({
  selector: 'app-portfolio-images-visiteur',
  templateUrl: './portfolio-images.component.html',
  styleUrls: ['./portfolio-images.component.css']
})
export class PortfolioImagesvisiteurComponent implements OnInit {
  tasks: Task[];
  maylikes: Task[]
  populars: Task[]
  categorie : string ; 
  subCategorie: string ; 
  searchText;
  search: string;
  p: number=1;

collection = [];
message:string;

  slideConfig = {"slidesToShow": 5, "slidesToScroll": 4};
  constructor(private price: PriceService , private resourceService:ProjetService ,private route:ActivatedRoute) {
    for (let i = 1; i <= 100; i++) {
      this.collection.push(`item ${i}`);
    }
  }


  ngOnInit(): void {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

   
   this.categorie = this.route.snapshot.paramMap.get('categorie') ; 
   this.subCategorie = this.route.snapshot.paramMap.get('subCategorie') ;
   if ((this.categorie) && !(this.subCategorie)) {
    
    this.getCategDyn(this.categorie )
    this.MostPopularcat (this.categorie)
    
   }
   else if((this.categorie) && (this.subCategorie)){
    
    
    this.getSubcategory(this.categorie , this.subCategorie)
    this.MostPopularsubcat (this.categorie,this.subCategorie)
    
   }
   else if(!(this.categorie) && !(this.subCategorie)){
    
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


  }
