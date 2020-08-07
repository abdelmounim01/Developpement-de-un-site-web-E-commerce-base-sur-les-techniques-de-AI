import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { ProjetService } from '../../../service/ImplementationService/projet.service'
import { Task } from 'src/app/service/Interfaces/task';
import { Metauser } from 'src/app/service/Interfaces/metauser';
import { AuthService } from 'src/app/service/auth/auth.service';
import { User } from 'src/app/service/Interfaces/user';
import { PriceService } from 'src/app/service/PriceService/price.service';
import  Swal  from 'sweetalert2';

@Component({
  selector: 'app-detail-product',
  templateUrl: './detail-product.component.html',
  styleUrls: ['./detail-product.component.css']
})
export class DetailProductComponent implements OnInit {
  id: number ;
  tasks: Task[];
  task: Task[]; 
  meta:Metauser[];
  Like : boolean = false ; 
  heartChange = 'far'
  rating:number
  starChangeOne = 'far'
  starChangeTwo = 'far'
  starChangeThree = 'far'
  starChangeFour = 'far'
  starChangeFive = 'far'
  user:User
  vendId:number
  UserId:number = JSON.parse(localStorage.getItem('user')).userId;
  idprod:number = +this.route.snapshot.paramMap.get('id')
  similiar: Task[]
  message:string;
  categorie:string
  subcategorie:string
  populars: Task[]

  constructor(private price: PriceService , private route:ActivatedRoute ,private resourceService:ProjetService,private authService: AuthService) { }
  slideConfig = {"slidesToShow": 5, "slidesToScroll": 4};
  ngOnInit() {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

    console.log(this.UserId)
    this.id = +this.route.snapshot.paramMap.get('id') ;
    console.log(this.id)
    this.getTask(this.id)
    console.log(this.tasks)
    this.similiarproducts(this.id)
    this.resourceService
      .getMetausers()
      .subscribe((data: Metauser[]) => {
        this.meta = data;
        this.meta.forEach(n => {
          if(n.produitId ==  this.id && n.userId == this.UserId  && n.like == 1)
          {
            this.heartChange = 'fas' ;
          }
          else if(n.produitId ==  this.id && n.userId == this.UserId  && n.rating == 1){
            this.starChangeOne = 'fas' ;
          }
          else if(n.produitId ==  this.id && n.userId == this.UserId  && n.rating == 2){
            this.starChangeTwo = 'fas' ;
          }
          else if(n.produitId ==  this.id && n.userId == this.UserId  && n.rating == 3){
            this.starChangeThree = 'fas' ;
          }
          else if(n.produitId ==  this.id && n.userId == this.UserId  && n.rating == 4){
            this.starChangeFour = 'fas' ;
          }
          else if(n.produitId ==  this.id && n.userId == this.UserId  && n.rating == 5){
            this.starChangeFive = 'fas' ;
          }
          
        });
  });
  
  this.resourceService
  .getTasks()
  .subscribe((data: Task[]) => {
    this.task = data;
    this.task.forEach(n => {
      if(n.produitId ==  this.idprod)
          {
            console.log("ok")
            this.vendId = n.vendeurId ;
            this.getUser(this.vendId)
          }
      });
  });
  this.resourceService
  .getTasks ()
  .subscribe((data: Task[]) => {
    this.task = data;
    this.task.forEach(n => {
      if(n.produitId ==  this.id)
          {
            this.categorie = n.Categories
            this.subcategorie = n.subCategory
            this.MostPopularsubcat(this.categorie,this.subcategorie);
            console.log(this.categorie)
            console.log(this.subcategorie)
          }
      });
  });
  console.log(this.vendId)
  }
  getLogin() {
    return this.authService.getUser().login;
  }

  logout() {
    return this.authService.logout();
  }

  getUser(vendId) : void{
    this.resourceService.getUser(this.vendId).subscribe(user => (this.user = user))
  }

  getTask (idprod): void {
    this.resourceService.getTask(idprod).subscribe(tasks => (this.tasks = tasks))
  }

  getMetausers (): void {
    this.resourceService.getMetausers().subscribe(meta => (this.meta = meta))
  }

  getMetauser (iduser): void {
    this.resourceService.getMetauser(iduser).subscribe(meta => (this.meta = meta))
  }

  addrating(idproduct:number,rating:number){
      if(rating == 1) {
        this.starChangeOne = 'fas'
        this.starChangeTwo = 'far'
        this.starChangeThree = 'far'
        this.starChangeFour = 'far'
        this.starChangeFive = 'far'
      }
      else if (rating == 2){
        this.starChangeOne = 'far'
        this.starChangeTwo = 'fas'
        this.starChangeThree = 'far'
        this.starChangeFour = 'far'
        this.starChangeFive = 'far'
      }
      else if (rating == 3){
        this.starChangeOne = 'far'
        this.starChangeTwo = 'far'
        this.starChangeThree = 'fas'
        this.starChangeFour = 'far'
        this.starChangeFive = 'far'
      }
      else if (rating == 4){
        this.starChangeOne = 'far'
        this.starChangeTwo = 'far'
        this.starChangeThree = 'far'
        this.starChangeFour = 'fas'
        this.starChangeFive = 'far'
      }
      else if (rating == 5){
        this.starChangeOne = 'far'
        this.starChangeTwo = 'far'
        this.starChangeThree = 'far'
        this.starChangeFour = 'far'
        this.starChangeFive = 'fas'
      }
      this.Like = !this.Like ;
      this.resourceService
      .getMetausers()
      .subscribe((data: Metauser[]) => {
      this.meta = data;
    const newMeta: Metauser = { userId:this.UserId , produitId : idproduct ,rating:rating} as Metauser
    this.resourceService.addrating(newMeta).subscribe(() => this.getMetausers())
    })
  }

  addpanier (idproduct:number): void {
    this.resourceService
    .getMetausers()
    .subscribe((data: Metauser[]) => {
      this.meta = data;
    const newMeta: Metauser = { userId:this.UserId , produitId : idproduct } as Metauser
    this.resourceService.addpanier(newMeta).subscribe(() => this.getMetausers())
    })
  }

  addlike(idproduct:number){
    if(this.heartChange == 'far'){
      this.heartChange = 'fas'
    } else {
    this.heartChange = 'far'

    }
    this.resourceService
    .getMetausers()
    .subscribe((data: Metauser[]) => {
      this.meta = data;
    const newMeta: Metauser = { userId:this.UserId , produitId : idproduct } as Metauser
    this.resourceService.addlike(newMeta).subscribe(() => this.getMetausers())
    })
    
   }

   somethingChanged(item){    
    var num1= item;
    console.log(num1);
}
similiarproducts(idprod):void{
  this.resourceService.similiarproducts(idprod).subscribe(similiar => (this.similiar = similiar))
}
MostPopularsubcat (categorie,subcategorie): void {
  this.resourceService.popularproductssubcat(categorie,subcategorie).subscribe(populars => (this.populars = populars))
}
Alert(){
  Swal.fire({
    icon: 'success',
    title: 'Gestion Panier',
    showConfirmButton: false,
    timer: 1500
  })
    
}
}

