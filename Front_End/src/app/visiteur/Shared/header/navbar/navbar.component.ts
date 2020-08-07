import { PriceService } from './../../../../service/PriceService/price.service';
import { Component, OnInit, Input, Output } from '@angular/core';
import { ProjetService } from '../../../../service/ImplementationService/projet.service'
import { Router } from '@angular/router'
import { User } from '../../../../service/Interfaces/user';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { EventEmitter } from 'protractor';

@Component({
  selector: 'app-navbar-visiteur',
  templateUrl: './navbar.component.html',
  styleUrls: ['./navbar.component.css']
})
export class NavbarvisiteurComponent implements OnInit {
  collapsed = true;
  exist:any
  users: User[]
  form: FormGroup;
  model: any = {};
  message:string;

  constructor(private price: PriceService , private _auth: ProjetService, private _router: Router,private formBuilder: FormBuilder) { }

  ngOnInit() {
    this.price.currentMessage.subscribe(message => this.message = message) ; 

    this.form = this.formBuilder.group({
      email: [null, [Validators.required, Validators.email]],
      password: [null, Validators.required],
    });
  }
  
  newMessageDollar() {
    this.price.changeMessage("Dollar")
  }
  newMessageDH() {
    this.price.changeMessage("DH")
  }
  newMessageEuro() {
    this.price.changeMessage("Euro")
  }


  loginUser (Email: string,Password:string) {
    var iduser:number
    this._auth
      .getUsers()
      .subscribe((data: User[]) => {
        this.users = data;
        this.users.forEach(n => {
          if(n.email == Email && n.password == Password )
          {
            iduser = n.userId
            console.log(n.type)
            if(n.type == 'Vendeur'){this.exist = 1}
            else if (n.type == 'Client'){this.exist = 2}
            else if(n.type == 'Admin'){this.exist = 3}
          } 
        });

        if(this.exist==1){
          alert("Welcome Vendeur");
          console.log('Tentative de connexion vendeur');
          // Vérifier que login/mdp sont correctes, par exemple par une requête à un service REST
          localStorage.setItem('user', JSON.stringify({email:Email,userId:iduser}));
          this._router.navigate(['/app-vendeur'])
        }
        else if (this.exist==2) {
          alert("Welcome Client");
          console.log('Tentative de connexion client');
          // Vérifier que login/mdp sont correctes, par exemple par une requête à un service REST
          localStorage.setItem('user', JSON.stringify({email:Email,userId:iduser}));
          this._router.navigate(['/Home'])
        } 
        else if (this.exist==3) {
          alert("Welcome Admin");
          console.log('Tentative de connexion admin');
          // Vérifier que login/mdp sont correctes, par exemple par une requête à un service REST
          localStorage.setItem('user', JSON.stringify({email:Email,userId:iduser}));
          this._router.navigate(['/app-admin'])
        }
        else{
          alert("Email or Password Incorrect");
        }
        this.exist = 0
      });
  }

  convert(x){
    /*
    console.log(x);
    
      if ( x == 'MAD'){
      this._auth.getTasks().subscribe(tasks => tasks.forEach(n => n.Product_price * 50))
       } else if( x == 'Euro') {
        this._auth.getTasks().subscribe(tasks => tasks.forEach(n => n.Product_price * 5))
       } else if (x == 'Dollar') {
        this._auth.getTasks().subscribe(tasks => tasks.forEach(n => n.Product_price * 100))
       }*/
  }

}
