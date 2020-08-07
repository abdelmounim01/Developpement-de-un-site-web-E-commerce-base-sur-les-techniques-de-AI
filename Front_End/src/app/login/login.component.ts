import { Component, OnInit } from '@angular/core';
import { ProjetService } from '../service/ImplementationService/projet.service'
import { Router } from '@angular/router'
import { User } from '../service/Interfaces/user';
import { FormGroup, FormBuilder, Validators } from '@angular/forms';
import { AlertsService } from 'angular-alert-module';

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css']
})
export class LoginComponent implements OnInit {

  exist:any
  users: User[]
  form: FormGroup;
  model: any = {};
  constructor(private _auth: ProjetService, private _router: Router,private formBuilder: FormBuilder ,  private alerts: AlertsService) { }

  ngOnInit() {
    this.form = this.formBuilder.group({
      email: [null, [Validators.required, Validators.email]],
      password: [null, Validators.required],
    });
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
          this.alerts.setMessage('Email or Password Incorrect !','error');    
          this.alerts.setDefaults('timeout',5);        }
        this.exist = 0
      });
      
      
      
    
  }

}
